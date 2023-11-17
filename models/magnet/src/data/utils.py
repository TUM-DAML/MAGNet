import bz2
from copy import deepcopy
from itertools import chain

import _pickle as cPickle
import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import torch
from networkx.algorithms import isomorphism
from torch.nn.utils.rnn import pad_sequence

from baselines.magnet.src.chemutils.constants import ATOM_LIST
from baselines.magnet.src.chemutils.hypergraph import MolDecomposition, is_all_cyclic
from baselines.magnet.src.chemutils.motif_helpers import atom_multiset_to_counts
from baselines.magnet.src.chemutils.rdkit_helpers import smiles_to_array


def split_array(arr, chunk_sizes, axis=0):
    """
    Splits a NumPy array into differently sized chunks along a specified dimension.

    Parameters:
        arr (numpy.ndarray): The input array.
        chunk_sizes (List[int]): A list of chunk sizes for each split.
        axis (int): The dimension along which to split the array (default: 0).

    Returns:
        List[numpy.ndarray]: A list of split arrays.
    """
    assert arr.ndim >= axis + 1, "Invalid axis specified"
    assert len(chunk_sizes) > 0, "No chunk sizes specified"

    arr_shape = list(arr.shape)
    total_size = arr_shape[axis]

    assert sum(chunk_sizes) == total_size, "Total chunk sizes don't match the input array size"

    indices = np.cumsum(chunk_sizes)[:-1]
    indices = np.concatenate((indices, [total_size]))

    return np.split(arr, indices, axis=axis)


def get_mol_decomposition(smiles, idx, mol_object_dir):
    if idx is None:
        decomposition = MolDecomposition(smiles)
    else:
        #  load preprocessed decomposition of molecule, including several molecule features
        data = bz2.BZ2File(mol_object_dir / f"{idx:06d}.pbz2", "rb")
        i, stored_smiles, decomposition = cPickle.load(data)
        assert i == idx
        assert smiles == stored_smiles
    return decomposition


def get_allowed_joins(_allowed_joins_dict, reverse_sort, nodes_in_shape):
    allowed_joins_dict = {
        j.item(): _allowed_joins_dict[i] for i, j in enumerate(reverse_sort) if i in _allowed_joins_dict.keys()
    }
    allowed_joins = [[allowed_joins_dict[n] for n in nis] for nis in nodes_in_shape]
    return list(chain(*allowed_joins))


def smiles_to_adjacency(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol)
    return adjacency


def get_gt_ordering(reference, input):
    if np.array_equal(reference, input):
        return np.arange(reference.shape[0])
    # test cyclic permutations in case order to traverse is unique
    idx_in = np.arange(reference.shape[0])
    for k in range(reference.shape[0]):
        idx = np.concatenate((idx_in[k:], idx_in[:k]))
        adj_res = input[idx, :][:, idx]
        if np.array_equal(reference, adj_res):
            return idx
    # resort to graph matching
    g1 = nx.from_numpy_array(reference)
    g2 = nx.from_numpy_array(input)
    gm = isomorphism.GraphMatcher(g2, g1)
    if not gm.is_isomorphic():
        # happens once in the dataset, we have to investigate this later
        # warnings.warn("GRAPH ISOMORPHISM FAILED!")
        return idx
    ordering = np.array(list(gm.mapping.keys()))
    return ordering


def prepare_motif_target(motifs, shape_node_idx, reference_dict, pp_orderings, atom_to_class):
    all_atom_idx, all_bond_types, all_charges = [], [], []
    for motif, s_idx in zip(motifs, shape_node_idx):
        reference_smiles = reference_dict[s_idx]
        atom_idx, bond_types, charges = get_canonical_motif_sequence(
            reference_smiles, motif, atom_to_class, pp_orderings, s_idx
        )
        all_atom_idx.append(atom_idx)
        all_bond_types.append(bond_types)
        all_charges.append(charges)
    return all_atom_idx, all_bond_types, all_charges


def get_type_and_num_joins(_mol):
    if is_all_cyclic(_mol):
        shape_type = "ring"
    elif max([atom.GetDegree() for atom in _mol.GetAtoms()]) > 2:
        shape_type = "junction"
    else:
        shape_type = "chain"

    # calculate number of available joins here
    if shape_type == "chain":
        atom_idxs = [a.GetIdx() for a in _mol.GetAtoms()]
        atom_idxs = [atom_idxs[0], atom_idxs[-1]]
    elif shape_type == "junction":
        atom_idxs = [a.GetIdx() for a in _mol.GetAtoms() if a.GetDegree() < 3]
    else:
        atom_idxs = [a.GetIdx() for a in _mol.GetAtoms()]
    return shape_type, atom_idxs


def get_canonical_motif_sequence(reference_smiles, input_smiles, atom_to_class, pp_orderings, s_idx):
    assert s_idx in pp_orderings.keys()
    if input_smiles in pp_orderings[s_idx].keys():
        return pp_orderings[s_idx][input_smiles]
    else:
        reference_adj = smiles_to_adjacency(reference_smiles)
        input_adj = smiles_to_adjacency(input_smiles)
        ordering = get_gt_ordering(reference_adj, input_adj)
        _, node_feats, edge_feats, _, charges = smiles_to_array(input_smiles, atom_to_class, return_charges=True)
        pp_orderings[s_idx][input_smiles] = (
            node_feats[ordering],
            edge_feats[ordering, :][:, ordering],
            charges[ordering],
        )
        return node_feats[ordering], edge_feats[ordering, :][:, ordering], charges[ordering]


def get_join_identity_adjacency(atom_idx, shape_adj, nodes_in_shape, num_nodes_in_shape):
    nodes_id_shapes = [i.tolist() for i in torch.split(nodes_in_shape.int(), num_nodes_in_shape)]
    shape_i, shape_j, _ = shape_adj.coo()
    target_vals, hypernode_idx = [], []
    for i, j in zip(shape_i, shape_j):
        shared_node_ids = set(nodes_id_shapes[i]) & set(nodes_id_shapes[j])
        assert len(shared_node_ids) == 1
        shared_node_ids = list(shared_node_ids)[0]
        target_vals.append(atom_idx[shared_node_ids] + 1)
        hypernode_idx.append(shared_node_ids)
    hypernode_ids = torch.tensor(target_vals)
    hypernode_idx = torch.tensor(hypernode_idx)
    hypernode_adj = deepcopy(shape_adj)
    hypernode_adj = hypernode_adj.set_value(hypernode_ids, layout="coo")

    # get per-shape atom counts for easier motif comparison
    num_hypernodes_in_shape = shape_adj.sum(-1).int().tolist()
    hypernode_counts = atom_multiset_to_counts(hypernode_ids, num_hypernodes_in_shape, len(ATOM_LIST) + 1)[:, 1:]
    return hypernode_adj, hypernode_counts, hypernode_idx


def prepare_leaf_target_seq(shape_classes, atom_idx, atom_adj_feats, nodes_in_shape, num_nodes_in_shape, tokens):
    leaf_mask = shape_classes[:, 0] == -1
    # each node can have one leaf at max
    has_leaf, leaf_idx, bond_type = atom_adj_feats[:, leaf_mask].coo()
    leaf_atom_class = atom_idx[leaf_mask][leaf_idx]
    atom_target = torch.zeros_like(atom_idx).long()
    # shift one to the left s.t. zero means no leaf node
    atom_target[has_leaf] = leaf_atom_class + 1
    # bond ground truth only for those atoms that actually has leafs
    bond_target = torch.zeros_like(atom_idx).long()
    bond_target[has_leaf] = bond_type

    atom_per_shape = torch.split(atom_target[nodes_in_shape.long()], num_nodes_in_shape)
    bond_per_shape = torch.split(bond_target[nodes_in_shape.long()], num_nodes_in_shape)
    # prepare tensors with padded target sequence
    atom_target_seq = pad_sequence(atom_per_shape, batch_first=True, padding_value=tokens["pad_token"]).long()
    bond_target_seq = pad_sequence(bond_per_shape, batch_first=True, padding_value=tokens["pad_token"]).long()
    return atom_target_seq.long(), bond_target_seq.long()


def get_atom_multiplicity_from_unsorted(atom_types):
    """
    Calculates positional tokens for a set of indices.

    Args:
        indices: A tensor of shape (n_indices,) containing indices.

    Returns:
        A tensor of shape (n_indices,) containing positional tokens.
    """
    max_elem = atom_types.max()
    counts = np.zeros(max_elem + 1)
    mults = []
    for at in atom_types:
        mults.append(counts[at])
        counts[at] += 1
    return mults


def get_atom_positional(atom_idx: torch.tensor, nodes_in_shape: np.array, num_nodes_in_shape: list):
    """
    Calculates positional tokens for each atom in a shape and positional tokens per node for each atom.

    Args:
        atom_indices: A tensor of shape (n_atoms,) containing indices of atoms.
        nodes_in_shape: An array of shape (n_nodes_in_shape,) containing indices of nodes in a shape.
        num_nodes_in_shape: An integer specifying the number of nodes in a shape.

    Returns:
        A tuple containing two tensors:
        - positional_tokens: A tensor of shape (n_atoms_in_shape,) containing positional tokens for each atom in a shape.
        - node_positional_tokens: A tensor of shape (n_atoms, 2) containing positional tokens per node for each atom,
          i.e., two entries (hypernode), one, or none (leaf).
    """
    atom_idx_in_shape = atom_idx[nodes_in_shape.astype(np.long)]
    atom_types_split = torch.split(atom_idx_in_shape, num_nodes_in_shape)
    mult_node_in_shape = np.concatenate([get_atom_multiplicity_from_unsorted(ats) for ats in atom_types_split])
    # get atom positional token per node, i.e. two entries (hypernode), one, or none (leaf)
    mult_per_atom = []
    for k in range(atom_idx.size(0)):
        mult = mult_node_in_shape[nodes_in_shape == k]
        mult_per_atom.append(mult.tolist() + [-1] * (3 - mult.size))
    mult_per_atom = torch.tensor(mult_per_atom)
    return mult_node_in_shape, mult_per_atom
