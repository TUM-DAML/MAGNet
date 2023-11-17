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

from models.global_utils import BASELINE_DIR
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


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--property", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="single")
    parser.add_argument("--ncpu", type=int, default=8)
    args = parser.parse_args()

    args.vocab = PairVocab(
        args.vocab,
        BASELINE_DIR=BASELINE_DIR,
        cuda=False,
        property=args.property,
    )

    pool = Pool(args.ncpu)
    random.seed(1)

    if args.mode == "single":
        # dataset contains single molecules
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize, vocab=args.vocab)

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

            with open(args.outdir + "tensors-%d.pkl" % split_id, "wb") as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == "pair":
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab=args.vocab)
        all_data = []
        for out in tqdm.tqdm(pool.imap_unordered(func, batches), total=len(batches)):
            all_data.append(out)
            pass
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open(args.outdir + "tensors-%d.pkl" % split_id, "wb") as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
