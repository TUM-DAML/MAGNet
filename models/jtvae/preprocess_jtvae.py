import pickle
from multiprocessing import Pool
from optparse import OptionParser

import rdkit
import tqdm

from models.jtvae.fastjt.mol_tree import MolTree
from models.global_utils import CKPT_DIR, SMILES_DIR, DATA_DIR
from pathlib import Path


def tensorize(input, assm=True):
    smiles, number = input
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    return mol_tree


def preprocess_jtvae(dataset_name: str, num_processes: int, num_splits: int = 10):
    train_path = SMILES_DIR / dataset_name / "train.txt"
    out_dir = DATA_DIR / "JTVAE" / dataset_name / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = Pool(num_processes)
    num_splits = int(num_splits)

    with open(train_path) as f:
        data = [(line.strip("\r\n ").split()[0], k) for k, line in enumerate(f)]

    all_data = []
    for out in tqdm.tqdm(pool.imap_unordered(tensorize, data), total=len(data)):
        all_data.append(out)
        pass

    le = int((len(all_data) + num_splits - 1) / num_splits)

    for split_id in range(num_splits):
        st = split_id * le
        print(st, le)
        sub_data = all_data[st : st + le]

        with open(out_dir / ("tensors-%d.pkl" % split_id), "wb") as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
