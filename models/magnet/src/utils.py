import math
import os
import pickle
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from rdkit import Chem
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from torch_sparse import SparseTensor

SMALL_INT = -math.log2(sys.maxsize * 2 + 2)
LARGE_INT = -SMALL_INT
NUM_STABLE = 1e-8


def calculate_balanced_acc(target, logits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = balanced_accuracy_score(target.cpu(), logits.argmax(-1).cpu())
    return accuracy


def calculate_class_weights(target, num_classes):
    one_hot_target = torch.nn.functional.one_hot(target.long(), num_classes=num_classes)
    weight = one_hot_target.sum() / one_hot_target.sum(0)
    return weight


def sort_ndarray_by_dim(values: List[List], names: List[str], sort_by: List[str], argsort=False):
    dtype = []
    for name, l in zip(names, values):
        if isinstance(l[0], str):
            dtype.append((name, np.unicode_, 64))
            continue
        dtype.append((name, type(l[0])))
    values = np.array(list(zip(*values)), dtype=dtype)
    if argsort:
        return np.argsort(values, order=sort_by, kind="stable")
    return np.sort(values, order=sort_by)


def manual_batch_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], Tensor) or isinstance(batch[key], SparseTensor):
            batch[key] = batch[key].to(device)


def mol_standardize(mol: Chem.Mol, largest_comp: bool):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True)
    if largest_comp:
        smiles = max(smiles.split("."), key=len)
    return smiles


def extract_blockdiag_sparse(matrix: sp.coo_matrix, block_sizes: Union[List, Tuple]):
    # Ensure matrix and block_sizes are valid inputs
    if len(block_sizes) == 1:
        return [matrix]

    # Calculate the cumulative sums of block sizes
    cumulative_sizes = np.cumsum(block_sizes)
    if cumulative_sizes[-1] != matrix.shape[0]:
        raise ValueError("Sum of block sizes does not match matrix size.")

    # Extract the blocks as a list of arrays
    blocks = []
    start = 0
    for size in block_sizes:
        end = start + size
        blocks.append(matrix[start:end, start:end])
        start = end

    return blocks


def smiles_from_file(file_path: Union[str, Path]):
    with open(file_path, "r") as file:
        smiles = file.readlines()
    smiles = [gt.strip("\n") for gt in smiles]
    assert all([Chem.MolFromSmiles(s) is not None for s in smiles])
    return smiles


def save_model_config_to_file(project_path, run_id, config_in, model):
    # save configs manually for easy reload later
    config = deepcopy(config_in)
    save_path = project_path / str(run_id) / "checkpoints"
    os.makedirs(save_path)
    save_path = save_path / "load_config.pkl"
    pop_keys = [key for key in config.keys() if key not in model.__dict__.keys()]
    pop_keys = [config.pop(key) for key in pop_keys]
    with open(save_path, "wb") as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
