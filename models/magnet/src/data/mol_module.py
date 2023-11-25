import os
import pickle
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import rdkit.Chem as Chem
import scipy.sparse as sp
import torch
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from scipy.sparse import block_diag
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from models.global_utils import SMILES_DIR, DATA_DIR
from models.magnet.src.chemutils.cangen import sort_sample
from models.magnet.src.chemutils.constants import ATOM_LIST, BOND_LIST
from models.magnet.src.chemutils.rdkit_helpers import (
    get_atom_charges,
    pred_to_mols,
    smiles_to_array,
)
from models.magnet.src.data.utils import *


class MolDataModule(pl.LightningDataModule):
    """
    Lightning Data Module to handle different datasets & loaders
    """

    def __init__(self, dataset, batch_size: int = 32, num_workers=None, shuffle=True):
        super().__init__()
        self.name = dataset
        self.shuffle = shuffle
        self.ef_size = len(BOND_LIST)
        self.dl_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": collate_fn,
        }

    def setup(self, stage=None):
        self.train_ds = MoleculeDataset(self.name, "train.txt")
        self.val_ds = MoleculeDataset(self.name, "val.txt")
        self.test_ds = MoleculeDataset(self.name, "test.txt")
        # make dimensions for model available
        self.feature_sizes = self.train_ds.feature_sizes

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=self.shuffle, **self.dl_args)

    def val_dataloader(self):
        if self.val_ds is not None:
            return DataLoader(self.val_ds, shuffle=False, **self.dl_args)
        return None

    def test_dataloader(self):
        if self.test_ds is not None:
            return DataLoader(self.test_ds, shuffle=False, **self.dl_args)
        return None


class MoleculeDataset(torch.utils.data.Dataset):
    """
    Torch Dataset to handle loading of individual molecules from one dataset
    """

    def __init__(self, dataset_name, filename):
        super().__init__()
        filepath = SMILES_DIR / dataset_name.lower() / filename
        with open(filepath, "rb") as file:
            self.all_smiles = file.readlines()
        self.mol_object_dir = DATA_DIR / "MAGNET" / dataset_name / filepath.stem

        self.atoms = ATOM_LIST
        self.atom_to_id, self.id_to_atom = dict(), dict()
        for i, a in enumerate(self.atoms):
            self.atom_to_id[a] = i
            self.id_to_atom[i] = a

        target_file = DATA_DIR / "MAGNET" / dataset_name / "magnet_vocab.pkl"
        with open(target_file, "rb") as file:
            shape_vocab = pickle.load(file)

        vocab_stats = shape_vocab.pop("stats")

        self.shapes = list(shape_vocab.keys())
        self.hash_to_id, self.id_to_hash, self.shape_to_size = dict(), dict(), dict()
        self.shape_reference, self.shape_type, self.shape_num_joins = dict(), dict(), dict()
        self.shape_to_join_idx, self.shape_allocs_to_gt_ordering = dict(), dict()
        for i, s in enumerate(self.shapes):
            self.hash_to_id[s] = i
            self.id_to_hash[i] = s
            _mol = Chem.MolFromSmiles(shape_vocab[s][0])
            self.shape_to_size[i] = _mol.GetNumAtoms()

            # preprocess orderings beforehand because graph matching is slow
            self.shape_allocs_to_gt_ordering[i] = dict()
            self.shape_reference[i] = shape_vocab[s][0]

            # get shape type and join idx, e.g. it is a ring and can be joined everywhere
            shape_type, join_idx = get_type_and_num_joins(_mol)
            self.shape_type[i] = shape_type
            self.shape_to_join_idx[i] = join_idx
            self.shape_num_joins[i] = len(join_idx)
        self.fingerprint_generator = MakeGenerator(("RDKit2D",))

        self.sequence_tokens = dict(start_token=-1, end_token=-2, pad_token=-3)
        # Prepare feature sizes and make accessible to model
        dummy_item = self.__getitem__(0)
        self.feature_sizes = dict(
            graph_feat_size=dummy_item["graph_features"].shape[-1],
            motif_feat_size=dummy_item["motif_features"].shape[-1] // 3,
            num_atoms=len(self.atoms),
            num_shapes=len(self.shapes),
            atom_adj_feat_size=len(BOND_LIST),
            min_size=vocab_stats["min_mol_size"],
            max_size=vocab_stats["max_mol_size"],
            # add additional buffer in case we have a molecule with more atoms than the max
            max_shape_size=vocab_stats["max_shape_size"] + 5,
            max_mult_shapes=vocab_stats["max_num_shapes"] + 5,
        )

    def __len__(self):
        return len(self.all_smiles)

    def __getitem__(self, idx):
        return self._getitem(idx, smiles=None)

    def _getitem(self, idx, smiles: str = None):
        smiles = self.all_smiles[idx].decode("utf-8").strip("\n") if smiles is None else smiles
        decomposition = get_mol_decomposition(smiles, idx, self.mol_object_dir)
        hypergraph_dict = decomposition.get_batch_output(self.hash_to_id)
        adj, atom_idx, bond_adj, mol = smiles_to_array(smiles, self.atom_to_id)
        atom_idx, adj, bond_adj, idx_sorted = sort_sample(mol, atom_idx, adj, bond_adj)
        reverse_sort = np.argsort(idx_sorted)
        nodes_in_shape = [reverse_sort[nis].tolist() for nis in hypergraph_dict["nodes_in_shape"]]
        # get sorted nodes in shape
        output_mols = pred_to_mols(
            dict(
                atom_idx=torch.tensor(atom_idx[list(chain(*nodes_in_shape))]),
                atom_adj=torch.tensor(bond_adj[list(chain(*nodes_in_shape))][:, list(chain(*nodes_in_shape))]),
                atom_charges=get_atom_charges(mol)[idx_sorted][list(chain(*nodes_in_shape))],
                num_nodes=[Chem.MolFromSmiles(m).GetNumAtoms() for m in hypergraph_dict["gt_motif_smiles"]],
            ),
            self.id_to_atom,
        )
        nodes_in_shape_new = []
        for k, o in enumerate(output_mols):
            Chem.MolToSmiles(o)
            rank = list(Chem.rdmolfiles.CanonicalRankAtoms(o))
            sort_idx = torch.sort(torch.tensor(rank))[1]
            nodes_in_shape_sorted = np.array(nodes_in_shape[k])[sort_idx]
            nodes_in_shape_new.append(nodes_in_shape_sorted.tolist())
        nodes_in_shape = nodes_in_shape_new
        allowed_joins = get_allowed_joins(hypergraph_dict["allowed_joins"], reverse_sort, nodes_in_shape)

        # collect motif sequence target
        motif_atom_target, motif_bond_target, motif_charge_target = prepare_motif_target(
            hypergraph_dict["gt_motif_smiles"],
            hypergraph_dict["hgraph_nodes"],
            self.shape_reference,
            self.shape_allocs_to_gt_ordering,
            self.atom_to_id,
        )
        # prepare atom positional tokens (relative to motif)
        array_nodes_in_shape = np.array(list(chain(*nodes_in_shape)))
        mult_node_in_shape, mult_per_atom = get_atom_positional(
            torch.tensor(atom_idx), array_nodes_in_shape, [len(n) for n in nodes_in_shape]
        )

        # prepare atom_idx to shape_idx map
        is_in_shape = (
            torch.arange(torch.tensor(atom_idx).size(0)).view(-1, 1) == torch.tensor(array_nodes_in_shape).int()
        )
        is_in_shape = torch.split(is_in_shape, [len(n) for n in nodes_in_shape], dim=1)
        is_in_shape = torch.cat([iis.sum(-1, keepdims=True) for iis in is_in_shape], dim=1)
        atom_to_shape_idx_map = [iis.nonzero().squeeze(-1).tolist() for iis in is_in_shape]

        return {
            "atom_adj": sp.coo_matrix(adj.astype(int)),
            "atom_idx": atom_idx,
            "bond_adj": sp.coo_matrix(bond_adj.astype(int)),
            "smiles": smiles,
            "graph_features": decomposition.fingerprint_mol,
            "shape_classes": hypergraph_dict["shape_classes"][idx_sorted],
            "motif_features": hypergraph_dict["motif_features"][idx_sorted],
            "shape_adj": sp.coo_matrix(hypergraph_dict["hgraph_adj"].astype(int)),
            "shape_nodes": hypergraph_dict["hgraph_nodes"],
            "shape_mult": hypergraph_dict["hgraph_nodes_mult"],
            "atom_charges": get_atom_charges(mol)[idx_sorted],
            "nodes_in_shape": nodes_in_shape,
            "motif_smiles": hypergraph_dict["gt_motif_smiles"],
            "allowed_joins": allowed_joins,
            "dataset_reference": self,
            "feats_per_motif": np.array([f.tolist() for f in hypergraph_dict["feats_per_motif"]]),
            "motif_atom_target": motif_atom_target,
            "motif_bond_target": [sp.coo_matrix(mb.astype(int)) for mb in motif_bond_target],
            "motif_charge_target": motif_charge_target,
            "mult_node_in_shape": mult_node_in_shape,
            "mult_per_atom": mult_per_atom,
            "atom_to_shape": atom_to_shape_idx_map,
        }


def collate_fn(batches, inference=False):
    """
    Take individual molecule graphs and merge into one, disconnected graph
    """
    atom_idx = torch.tensor(np.concatenate([b["atom_idx"] for b in batches], axis=0))
    # record number of nodes to restore samples from batch later
    num_nodes = [b["atom_adj"].shape[0] for b in batches]
    # create atom adjacency and bond adjacency for disconnected graph
    atom_adj = block_diag([b["atom_adj"] for b in batches])
    edge_index = np.stack((atom_adj.row, atom_adj.col))
    atom_adj = SparseTensor.from_edge_index(torch.tensor(edge_index).long(), torch.tensor(atom_adj.data))
    # stack other node features indicators
    shape_classes = torch.tensor(np.stack(list(chain(*[b["shape_classes"] for b in batches]))))
    atom_charges = torch.tensor(list(chain(*[b["atom_charges"] for b in batches])))
    motif_features = torch.tensor(np.stack(list(chain(*[b["motif_features"] for b in batches]))))
    # prepare atom positional tokens (relative to motif)
    mult_node_in_shape = np.concatenate([b["mult_node_in_shape"] for b in batches])
    mult_per_atom = np.concatenate([b["mult_per_atom"] for b in batches])
    if inference:
        return dict(
            atom_idx=atom_idx,
            atom_adj=atom_adj,
            shape_classes=shape_classes,
            atom_charges=atom_charges,
            motif_features=motif_features,
            mult_node_in_shape=torch.tensor(mult_node_in_shape),
            mult_per_atom=torch.tensor(mult_per_atom),
            num_nodes=num_nodes,
        )

    bond_adj = block_diag([b["bond_adj"] for b in batches])
    edge_index = np.stack((bond_adj.row, bond_adj.col))
    bond_adj = SparseTensor.from_edge_index(torch.tensor(edge_index).long(), torch.tensor(bond_adj.data))
    assert atom_adj.sizes() == bond_adj.sizes()
    assert torch.equal(bond_adj.coo()[0], atom_adj.coo()[0])
    assert torch.equal(bond_adj.coo()[1], atom_adj.coo()[1])
    # stack also the shape adjacency and prepare shape graph node sequence
    dataset_reference = batches[0]["dataset_reference"]
    tokens = dataset_reference.sequence_tokens
    shape_nodes_seq = [
        torch.tensor([tokens["start_token"]] + b["shape_nodes"] + [tokens["end_token"]], dtype=torch.long)
        for b in batches
    ]
    shape_nodes_seq = pad_sequence(shape_nodes_seq, batch_first=True, padding_value=tokens["pad_token"]).long()
    shape_node_idx = torch.tensor(np.concatenate([b["shape_nodes"] for b in batches], axis=0)).long()
    shape_node_mult = torch.tensor(np.concatenate([b["shape_mult"] for b in batches], axis=0))
    num_shape_nodes = [b["shape_adj"].shape[0] for b in batches]
    # stack node features in feature dimension
    # The identity of the shape corresponds to what is stored in b["shape_nodes"]!
    nodes_in_shape, atom_in_shape_mult = torch.Tensor([]), np.array([])
    counter, num_nodes_in_shape = 0, []
    for b in batches:
        for nps in b["nodes_in_shape"]:
            nis = torch.Tensor(nps) + counter
            nodes_in_shape = torch.cat((nodes_in_shape, nis))
            num_nodes_in_shape.append(len(nps))
            atom_in_shape_mult = np.concatenate(
                (atom_in_shape_mult, get_atom_multiplicity_from_unsorted(atom_idx[nis.long()]))
            )
        counter += b["atom_adj"].shape[0]

    # prepare atom_idx to shape_idx map
    ats_map_batches = list(chain(*[b["atom_to_shape"] for b in batches]))
    num_shape_nodes_ext = np.repeat(np.cumsum([0] + num_shape_nodes[:-1]), num_nodes)
    atom_to_shape_idx_map = [(np.array(t) + nn).tolist() for t, nn in zip(ats_map_batches, num_shape_nodes_ext)]

    allowed_joins = torch.tensor(list(chain(*[b["allowed_joins"] for b in batches])))
    shape_adj = block_diag([b["shape_adj"] for b in batches])
    edge_index = np.stack((shape_adj.row, shape_adj.col))
    shape_adj = SparseTensor.from_edge_index(
        torch.tensor(edge_index).long(), torch.tensor(shape_adj.data), sparse_sizes=shape_adj.shape
    )

    # stack rdkit features attained per graph
    graph_features = torch.stack([torch.tensor(b["graph_features"]) for b in batches])
    # if we only have a single sample we will only have a single list of
    # targets and not, as expected, a list of lists
    smiles = [b["smiles"] for b in batches]
    motif_smiles = list(chain(*[b["motif_smiles"] for b in batches]))
    # prepare leaf target sequence
    leaf_target = prepare_leaf_target_seq(shape_classes, atom_idx, bond_adj, nodes_in_shape, num_nodes_in_shape, tokens)
    # prepare features per motif
    feats_in_motif = torch.tensor(np.concatenate([b["feats_per_motif"] for b in batches], axis=0))
    # collect number of core node mols where hypernodes are counted twice!
    num_core_atoms_pre_join = [n.sum().item() for n in torch.split(torch.tensor(num_nodes_in_shape), num_shape_nodes)]
    # prepare join identity adjacency
    hypernode_adj, hypernode_counts_in_shape, join_idxs = get_join_identity_adjacency(
        atom_idx, shape_adj, nodes_in_shape, num_nodes_in_shape
    )
    # prepare motif targets
    motif_bond_target = block_diag(list(chain(*[b["motif_bond_target"] for b in batches])))
    edge_index = np.stack((motif_bond_target.row, motif_bond_target.col))
    motif_bond_target = SparseTensor.from_edge_index(
        torch.tensor(edge_index).long(), torch.tensor(motif_bond_target.data)
    )
    motif_atom_target = list(chain(*[b["motif_atom_target"] for b in batches]))
    motif_atom_target = [torch.tensor(mat) for mat in motif_atom_target]
    motif_atoms = torch.cat(motif_atom_target)
    motif_atom_target = pad_sequence(motif_atom_target, batch_first=True, padding_value=tokens["pad_token"]).long()
    motif_charge_target = list(chain(*[b["motif_charge_target"] for b in batches]))
    motif_charges = torch.cat(motif_charge_target)
    motif_charge_target = pad_sequence(motif_charge_target, batch_first=True, padding_value=tokens["pad_token"]).long()
    batch = {
        "atom_adj": atom_adj,
        "atom_idx": atom_idx,
        "bond_adj": bond_adj,
        "graph_features": graph_features,
        "num_nodes": num_nodes,
        "smiles": smiles,
        "motif_features": motif_features.float(),
        "motif_smiles": motif_smiles,
        "shape_classes": shape_classes,
        "shape_adj": shape_adj,
        "atom_charges": atom_charges,
        "shape_nodes_seq": shape_nodes_seq,
        "shape_node_idx": shape_node_idx,
        "shape_node_mult": shape_node_mult.long(),
        "num_nodes_hgraph": torch.tensor(num_shape_nodes, dtype=torch.int),
        "nodes_in_shape": nodes_in_shape.long(),
        "num_nodes_in_shape": torch.tensor(num_nodes_in_shape, dtype=torch.int),
        "allowed_joins": allowed_joins,
        "leaf_target": leaf_target,
        "feats_in_motif": feats_in_motif.float(),
        "atom_in_shape_mult": torch.tensor(atom_in_shape_mult).long(),
        "num_core_atoms_pre_join": torch.tensor(num_core_atoms_pre_join, dtype=torch.int),
        "mult_node_in_shape": torch.tensor(mult_node_in_shape),
        "mult_per_atom": torch.tensor(mult_per_atom),
        "hypernode_adj": hypernode_adj,
        "hypernode_counts_in_shape": hypernode_counts_in_shape,
        "motif_bond_target": motif_bond_target,
        "motif_atom_target": motif_atom_target,
        "motif_charge_target": motif_charge_target,
        "motif_atoms": motif_atoms,
        "motif_charges": motif_charges,
        "join_idxs": join_idxs,
        "atom_to_shape_idx_map": atom_to_shape_idx_map,
    }
    return batch
