import argparse
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy
import rdkit
import torch
import tqdm

from models.global_utils import BASELINE_DIR, SMILES_DIR, DATA_DIR
from models.hiervae.src.mol_graph import MolGraph
from models.hiervae.src.vocab import PairVocab, common_atom_vocab


def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y)  # no need of order for x


def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(",")) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,)  # no need of order for x

def preprocess_hiervae(dataset_name: str, num_processes: int, batch_size: int = 32):
    train_path = SMILES_DIR / dataset_name / "train.txt"
    out_dir = DATA_DIR / "HIERVAE" / dataset_name / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = PairVocab(
        dataset_name,
        BASELINE_DIR=BASELINE_DIR,
        cuda=False,
        property=None,
    )

    pool = Pool(num_processes)
    random.seed(1)

    # dataset contains single molecules
    with open(train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    random.shuffle(data)

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    func = partial(tensorize, vocab=vocab)

    all_data = []
    for out in tqdm.tqdm(pool.imap_unordered(func, batches), total=len(batches)):
        all_data.append(out)
        pass

    num_splits = len(all_data) // 100
    print(len(all_data), num_splits - 1, num_splits)
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open(out_dir / ("tensors-%d.pkl" % split_id), "wb") as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
