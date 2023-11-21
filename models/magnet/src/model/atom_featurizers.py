import copy
import torch
from torch.nn import Embedding
from torch_geometric.nn import MLP


class AtomFeaturizer(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        dim_config = parent_ref.dim_config
        feat_size = parent_ref.feature_sizes
        self.feat_size = feat_size
        self.output_dim_join = (
            dim_config["atom_id_dim"]
            + dim_config["motif_feat_dim"]
            + dim_config["atom_charge_dim"]
            + dim_config["shape_id_dim"]
            + dim_config["shape_multiplicity_dim"]
        )
        self.output_dim = (
            self.output_dim_join
            + dim_config["shape_id_dim"] * 2
            + dim_config["shape_multiplicity_dim"] * 2
            + dim_config["motif_feat_dim"] * 2
        )
        self.atom_id_emb = Embedding(feat_size["num_atoms"], dim_config["atom_id_dim"])
        self.atom_charge_emb = Embedding(3, dim_config["atom_charge_dim"])
        self.shape_id_emb = Embedding(feat_size["num_shapes"] + 1, dim_config["shape_id_dim"])
        self.motif_feat_transform = MLP(
            [feat_size["motif_feat_size"], dim_config["motif_feat_dim"]],
            act="relu",
            plain_last=True,
        )
        # we define featurizers here even though they are not used in this module
        # but this way, all embeddings are collected in the same place
        self.atom_mult_emb = Embedding(feat_size["max_shape_size"], dim_config["atom_multiplicity_dim"])
        self.shape_mult_emb = Embedding(feat_size["max_mult_shapes"], dim_config["shape_multiplicity_dim"])

    def forward(self, feat_dict):
        num_joins = 3
        atom_id_embs = self.atom_id_emb(feat_dict["atom_idx"])
        atom_charge_embs = self.atom_charge_emb(feat_dict["atom_charges"] + 1)
        mfeats = feat_dict["motif_features"]
        motif_feat_embs = self.motif_feat_transform(mfeats.view(-1, mfeats.shape[1] // num_joins))
        motif_feat_embs = motif_feat_embs.view(mfeats.shape[0], -1)
        shape_embs = [self.shape_id_emb(feat_dict["shape_classes"][:, j].long() + 1) for j in range(num_joins)]
        mult_embs = [self.atom_mult_emb(feat_dict["mult_per_atom"][:, j].long() + 1) for j in range(num_joins)]
        node_feats = torch.cat(
            [atom_id_embs, atom_charge_embs, motif_feat_embs, *shape_embs, *mult_embs],
            dim=-1,
        )
        return dict(node_feats=node_feats)


def sort_join_atoms(feat_dict, feat_size):
    # make sure everything is sorted as we expect it
    shape_classes_copy = copy.deepcopy(feat_dict["shape_classes"])
    max_num_shapes = feat_size["num_shapes"] + 1
    shape_classes_copy[shape_classes_copy == -1] = max_num_shapes
    shape_classes_sorted, sorted_idx = torch.sort(shape_classes_copy, dim=-1)
    shape_classes_sorted[shape_classes_sorted == max_num_shapes] = -1

    mshape = feat_dict["motif_features"].shape
    motif_features = feat_dict["motif_features"].view(-1, 2, mshape[1] // 2)
    motif_features_sorted = torch.zeros_like(motif_features)
    mult_sorted = torch.zeros_like(feat_dict["mult_per_atom"])
    for i, sidx in enumerate(sorted_idx):
        motif_features_sorted[i] = torch.index_select(motif_features[i], 0, sidx)
        mult_sorted[i] = torch.index_select(feat_dict["mult_per_atom"][i], 0, sidx)
    return shape_classes_sorted, motif_features_sorted, mult_sorted
