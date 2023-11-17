from copy import deepcopy

import networkx as nx
import torch
import torch_sparse
from torch_geometric.nn import MLP
from torch_sparse import SparseTensor

from models.magnet.src.chemutils.motif_helpers import atom_multiset_to_counts
from models.magnet.src.chemutils.rdkit_helpers import extract_single_graphs
from models.magnet.src.utils import SMALL_INT, calculate_class_weights


class ShapeConnectivityPredictor(torch.nn.Module):
    def __init__(self, parent_ref) -> None:
        super().__init__()
        dim_config = parent_ref.dim_config
        feature_sizes = parent_ref.feature_sizes
        feature_dim = (dim_config["shape_id_dim"] + dim_config["shape_multiplicity_dim"]) * 3 + dim_config["latent_dim"]
        self.num_atoms = 1 + feature_sizes["num_atoms"]
        self.mlp = MLP([feature_dim, 2 * feature_dim, feature_dim, self.num_atoms])
        self.node_aggregation = parent_ref.layer_config["node_aggregation"]

    def forward(self, batch, z_graph, sample, parent_ref):
        if sample:
            hypergraph_outputs, batch = self.forward_inference(batch, z_graph, parent_ref)
        else:
            hypergraph_outputs = self.forward_train(batch, z_graph, parent_ref)
        return hypergraph_outputs, batch

    def forward_inference(self, batch, z_graph, parent_ref):
        dataset = parent_ref.dataset_ref
        connectivity_logits = self.forward_train(batch, z_graph, parent_ref)["connectivity_logits"]
        connectivity_discrete, connectivity_confidence = self.cont_to_disc(batch, connectivity_logits, parent_ref)
        batch["shape_adj"] = SparseTensor.from_dense(connectivity_discrete)
        batch["shape_adj_logits"] = connectivity_logits

        batch["shape_adj"] = correct_shape_adjacency(connectivity_confidence, batch, dataset)
        rows, cols = batch["shape_adj"].nonzero().T
        hypernode_join = connectivity_discrete[rows, cols]
        batch["shape_adj"] = SparseTensor.from_dense(batch["shape_adj"])
        batch["hypernode_adj"] = deepcopy(batch["shape_adj"]).set_value(hypernode_join, layout="coo")

        num_hypernodes_in_shape = batch["shape_adj"].sum(-1).int().tolist()
        hypernode_counts = atom_multiset_to_counts(hypernode_join, num_hypernodes_in_shape, self.num_atoms)[:, 1:]
        batch["hypernode_counts_in_shape"] = hypernode_counts
        return dict(), batch

    def forward_train(self, batch, z_graph, parent_ref):
        dev = z_graph.device
        # prepare allowed edges, i.e. no cross-graph edges
        block_diag_edges = torch.block_diag(*[torch.ones((nn, nn)) for nn in batch["num_nodes_hgraph"]])
        edge_index = block_diag_edges.nonzero().to(dev)
        # featurize nodes with shape id as well as multiplicity
        shape_id_embedding = parent_ref.node_featurizer.shape_id_emb(batch["shape_node_idx"])
        shape_mult_embedding = parent_ref.node_featurizer.shape_mult_emb(batch["shape_node_mult"])
        x = torch.cat([shape_id_embedding, shape_mult_embedding], dim=-1)
        # aggregate over shape set to contextualize shape nodes
        shape_set_emb_split = torch.split(x, batch["num_nodes_hgraph"].tolist(), dim=0)
        agg_fn = getattr(torch, self.node_aggregation)
        shape_set_emb_agg = torch.stack([agg_fn(j, dim=0) for j in shape_set_emb_split])
        # predict edge existence
        num_shape_connections = batch["num_nodes_hgraph"] ** 2
        shape_set_emb_repeated = torch.repeat_interleave(shape_set_emb_agg, num_shape_connections, dim=0)
        z_graph_repeated = torch.repeat_interleave(z_graph, num_shape_connections, dim=0)
        input = torch.cat((x[edge_index].flatten(1), z_graph_repeated, shape_set_emb_repeated), dim=1)
        hypernode_join = self.mlp(input).squeeze(0)
        # enfore symmetric adjacency
        hypernode_join = SparseTensor.from_edge_index(edge_index.T, hypernode_join)
        hypernode_join = _to_symmetric(hypernode_join)
        return dict(connectivity_logits=hypernode_join)

    def cont_to_disc(self, batch, adj_continuous, parent_ref):
        adj_discrete = adj_continuous.clone()
        row, col, discrete_val = adj_discrete.coo()
        edge_index = torch.stack((row, col)).T
        discrete_val = torch.softmax(discrete_val, dim=-1)
        confidence, join_type = discrete_val.max(-1)
        threshold_mask = torch.logical_and((1 - discrete_val[:, 0]) > parent_ref.sample_threshold, join_type > 0)
        confidence = confidence[threshold_mask]
        join_type = join_type[threshold_mask]
        edge_index = edge_index[threshold_mask]
        sparse_size = (adj_discrete.size(0), adj_discrete.size(1))
        adj_discrete = SparseTensor.from_edge_index(edge_index.T, join_type, sparse_sizes=sparse_size)
        adj_confidence = SparseTensor.from_edge_index(edge_index.T, confidence, sparse_sizes=sparse_size)
        return adj_discrete.to_dense().squeeze(), adj_confidence.to_dense().squeeze()

    def calculate_loss(self, batch, decoder_outputs, dataset):
        _, _, hypernode_logits = decoder_outputs["connectivity_logits"].coo()
        hypernode_adj = batch["hypernode_adj"]
        hypernode_targets = extract_single_graphs(batch["num_nodes_hgraph"], hypernode_adj, flatten=True)
        weight = calculate_class_weights(hypernode_targets, self.num_atoms)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss_hypernode_joins = loss_fn(hypernode_logits, hypernode_targets)
        return dict(loss_shapeadj=loss_hypernode_joins)


def _to_symmetric(sparse_tensor):
    sparse_tensor = torch_sparse.cat([sparse_tensor, sparse_tensor.t()], dim=-1)
    _, _, vals = sparse_tensor.coo()
    vals = vals.view(vals.size(0), 2, -1).mean(-2)
    sparse_tensor = sparse_tensor.set_value(vals, layout="coo")
    return sparse_tensor


def correct_shape_adjacency(connectivity_confidence, batch, dataset):
    device = batch["shape_node_idx"].device
    ii, jj, confidence = SparseTensor.from_dense(torch.triu(connectivity_confidence)).coo()
    # sort edges to-be-added by confidence of prediction
    _, sorted_idx = torch.sort(confidence)
    ii = ii[sorted_idx]
    jj = jj[sorted_idx]

    shape_graph = nx.Graph()
    for k in range(connectivity_confidence.size(0)):
        shape_graph.add_node(k)
    shape_ids = batch["shape_node_idx"].tolist()

    for k, (i, j) in enumerate(zip(ii.tolist(), jj.tolist())):
        if i == j:
            continue
        type_i = dataset.shape_type[shape_ids[i]]
        type_j = dataset.shape_type[shape_ids[j]]
        # no two chains can be connected, rest is allowed
        if (type_i == "chain") and (type_j == "chain"):
            continue
        # respect maximum "available joins" of type
        num_available_joins_i = dataset.shape_num_joins[shape_ids[i]]
        if shape_graph.degree[i] >= num_available_joins_i:
            continue
        num_available_joins_j = dataset.shape_num_joins[shape_ids[j]]
        if shape_graph.degree[j] >= num_available_joins_j:
            continue
        # no more than 4 joins per shape
        if shape_graph.degree[i] >= 4 or shape_graph.degree[j] >= 4:
            continue
        # no cycles are allowed in the shape adjacency
        if not nx.has_path(shape_graph, i, j):
            shape_graph.add_edge(i, j)
        else:
            pass
    return torch.from_numpy(nx.to_numpy_array(shape_graph)).int().to(device)
