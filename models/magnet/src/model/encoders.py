import warnings

import numpy as np
import torch
import torch_geometric
from torch.nn import Embedding
from torch_geometric.nn import MLP, TransformerConv

from models.magnet.src.model.atom_featurizers import AtomFeaturizer


class GraphEncoder(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        # define encoder edge features
        node_feat_dim = parent_ref.node_featurizer.output_dim
        edge_hidden = node_feat_dim // 4
        dim_config = parent_ref.dim_config
        self.init_edges_enc = Embedding(parent_ref.feature_sizes["atom_adj_feat_size"], edge_hidden)
        self.node_aggregation = parent_ref.layer_config["node_aggregation"]

        self.enc_layers, self.shape_layers = [], []
        encoder_conv = TransformerConv(node_feat_dim, node_feat_dim, edge_dim=edge_hidden)
        encoder_func_def = "x, edge_index, edge_attr -> x"
        for _ in range(parent_ref.layer_config["num_layers_enc"]):
            self.enc_layers.extend([(encoder_conv, encoder_func_def), torch.nn.ReLU()])
        self.enc_layers = torch_geometric.nn.Sequential("x, edge_index, edge_attr", self.enc_layers)
        self.mlp_atoms = MLP([node_feat_dim, dim_config["enc_atom_dim"] * 2])

        # res connection
        node_feat_dim = node_feat_dim * 2
        shape_conv = TransformerConv(node_feat_dim, node_feat_dim, edge_dim=node_feat_dim * 3)
        for _ in range(parent_ref.layer_config["num_layers_shape_enc"]):
            self.shape_layers.extend([(shape_conv, encoder_func_def), torch.nn.ReLU()])
        self.shape_layers = torch_geometric.nn.Sequential("x, edge_index, edge_attr", self.shape_layers)

        summed_enc_dims = (
            dim_config["enc_atom_dim"]
            + dim_config["enc_shapes_dim"]
            + dim_config["enc_joins_dim"]
            + dim_config["enc_leafs_dim"]
            + dim_config["enc_global_dim"]
        )
        if not summed_enc_dims == dim_config["latent_dim"]:
            warnings.warn("Encoder dimensions do not add up to latent dimension")
        # add x2 because of "residual connection"
        self.mlp_shapes = MLP([node_feat_dim * 2, dim_config["enc_shapes_dim"] * 2])
        # input: 2x shape node embedding (of those that are joined) + join node embedding
        self.mlp_joins = MLP([node_feat_dim * 3, dim_config["enc_joins_dim"] * 2])
        # input: primary shape, secondary shape, attachement point, leaf node
        self.mlp_leafs = MLP([node_feat_dim * 4, dim_config["enc_leafs_dim"] * 2])
        self.mlp_global = MLP([parent_ref.feature_sizes["graph_feat_size"], dim_config["enc_global_dim"] * 2])

    def forward(self, batch: dict, node_featurizer: AtomFeaturizer):
        bs = len(batch["smiles"])
        node_feats = node_featurizer(batch)["node_feats"]
        row, col, _ = batch["atom_adj"].coo()
        edge_index = torch.stack([row, col], dim=0)

        _, _, bond_types = batch["bond_adj"].coo()
        # get class descriptors for embedding
        bond_types = bond_types - 1
        bond_types = self.init_edges_enc(bond_types)

        # transform embedding through MP layers
        node_embeddings = self.enc_layers(node_feats, edge_index, bond_types)
        agg_fn = getattr(torch, self.node_aggregation)

        # aggreate node embeddings per molecule
        nodes_in_mol = torch.split(node_embeddings, batch["num_nodes"])
        num_shape_nodes = batch["num_nodes_hgraph"].int().tolist()
        agg_node_embeddings = torch.stack([agg_fn(e, dim=0) for e in nodes_in_mol])
        latent_node_emb = self.mlp_atoms(agg_node_embeddings).view(bs, 2, -1)

        # from here, implement "residual connection" over GNN
        node_embeddings = torch.cat([node_feats, node_embeddings], dim=1)

        # transform global graph features
        latent_global_emb = self.mlp_global(batch["graph_features"]).view(bs, 2, -1)

        # aggregate node embeddings per shape
        nodes_in_shape = torch.split(batch["nodes_in_shape"], batch["num_nodes_in_shape"].tolist())
        shape_emb = [agg_fn(node_embeddings[idx.long()], dim=0) for idx in nodes_in_shape]
        shape_emb = torch.stack(shape_emb)

        # # create join embeddings by aggregating over connected-shape embeddings and join node embeddings
        row, col, _ = batch["hypernode_adj"].coo()
        shape_node_emb = torch.cat([shape_emb[row.long()], shape_emb[col.long()]], dim=1)
        join_node_emb = node_embeddings[batch["join_idxs"].long()]
        join_emb = torch.cat([shape_node_emb, join_node_emb], dim=1)
        num_joins = [nj.sum().int().item() for nj in torch.split(batch["shape_adj"].sum(-1), num_shape_nodes)]
        join_emb_agg = torch.stack([agg_fn(j, dim=0) for j in torch.split(join_emb, num_joins)])
        latent_join_emb = self.mlp_joins(join_emb_agg).view(bs, 2, -1)

        # create leaf node embeddings by aggregating over connected-shape embeddings and leaf node embeddings
        is_leaf_mask = batch["shape_classes"][:, 0] == -1
        num_leafs = [l.sum().int().item() for l in torch.split(is_leaf_mask, batch["num_nodes"])]
        # case: no leaf nodes in any of the samples of the batch
        if not is_leaf_mask.any():
            emb_size = node_embeddings.shape[-1] * 2 + shape_emb.shape[-1] * 2
            leaf_emb_agg = torch.zeros(bs, emb_size).to(node_embeddings.device)
        else:
            leaf_node_emb = node_embeddings[is_leaf_mask]
            attachement_idx = batch["atom_adj"][is_leaf_mask].coo()[1]
            attachement_emb = node_embeddings[attachement_idx]
            atom_to_shape_idx = np.array(batch["atom_to_shape_idx_map"], dtype=object)[attachement_idx.int().cpu()]
            # case: only one leaf present in batch
            if len(attachement_idx) == 1:
                atom_to_shape_idx = [atom_to_shape_idx]
            prim_shape_emb = torch.stack([shape_emb[a[0]] for a in atom_to_shape_idx])
            filler_emb = torch.zeros_like(prim_shape_emb[0])
            sec_shape_emb = torch.stack([shape_emb[a[1]] if len(a) == 2 else filler_emb for a in atom_to_shape_idx])
            leaf_emb = torch.cat([leaf_node_emb, attachement_emb, prim_shape_emb, sec_shape_emb], dim=1)
            leaf_emb_agg = torch.stack([agg_fn(l, dim=0) for l in torch.split(leaf_emb, num_leafs)])
        latent_leaf_emb = self.mlp_leafs(leaf_emb_agg).view(bs, 2, -1)

        # create meaningful shape embeddings through message passing on the shape adjacency
        row, col, _ = batch["shape_adj"].coo()
        shape_edge_index = torch.stack([row, col], dim=0)
        shape_embeddings = self.shape_layers(shape_emb, shape_edge_index, join_emb)
        shape_embeddings = torch.cat([shape_emb, shape_embeddings], dim=1)
        shape_emb_agg = torch.stack([agg_fn(s, dim=0) for s in torch.split(shape_embeddings, num_shape_nodes)])
        latent_shape_emb = self.mlp_shapes(shape_emb_agg).view(bs, 2, -1)

        graph_embedding = torch.cat(
            [
                latent_global_emb[:, 0],
                latent_node_emb[:, 0],
                latent_shape_emb[:, 0],
                latent_join_emb[:, 0],
                latent_leaf_emb[:, 0],
                latent_global_emb[:, 1],
                latent_node_emb[:, 1],
                latent_shape_emb[:, 1],
                latent_join_emb[:, 1],
                latent_leaf_emb[:, 1],
            ],
            dim=1,
        )
        return dict(z_graph_encoder=graph_embedding)
