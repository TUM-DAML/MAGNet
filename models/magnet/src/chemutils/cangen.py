import os

import numpy as np
import rdkit.Chem as Chem
import torch

from models.magnet.src.utils import sort_ndarray_by_dim


def sort_sample(mol, atom_idx, adj, edge_feats):
    _ = Chem.MolToSmiles(mol)
    canon_atom_position = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
    sorted_idx = np.argsort(canon_atom_position)
    values = [atom_idx.tolist(), sorted_idx.tolist()]
    names = ["atom_idx", "canon_atom_order"]
    # IMPORTANT: this sorting needs to be "stable" to retain in-motif atom ordering
    idx_sorted = sort_ndarray_by_dim(values, names=names, sort_by=names, argsort=True).astype(np.long)
    atom_idx = atom_idx[idx_sorted]
    adj = adj[idx_sorted[None, :], idx_sorted[:, None]]
    edge_feats = edge_feats[idx_sorted[None, :], idx_sorted[:, None]]
    return atom_idx, adj, edge_feats, idx_sorted


def sort_with_dupes(input):
    assert isinstance(input, torch.Tensor)
    sorted_list, idx = torch.sort(input)
    counter = 0
    idx_dupes = []
    for i, e in enumerate(sorted_list):
        if i == 0:
            idx_dupes.append(counter)
        else:
            if e > sorted_list[i - 1]:
                counter += 1
            idx_dupes.append(counter)
    _, inverse = torch.sort(idx)
    return torch.Tensor(idx_dupes)[inverse]


def twod_sort_with_dupes(input):
    # assert isinstance(input, np.array)
    idx = np.argsort(input, order=("x", "y"))
    sorted_list = input[idx]
    counter = 0
    idx_dupes = []
    for i, e in enumerate(sorted_list):
        if i == 0:
            idx_dupes.append(counter)
        else:
            if e[0] > sorted_list[i - 1][0] or e[1] > sorted_list[i - 1][1]:
                counter += 1
            idx_dupes.append(counter)
    inverse = np.argsort(idx)
    return np.array(idx_dupes)[inverse]
