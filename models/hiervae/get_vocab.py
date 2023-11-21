import argparse
import sys
from multiprocessing import Pool

from rdkit import Chem

from models.hiervae.src.mol_graph import MolGraph
from models.global_utils import BASELINE_DIR, SMILES_DIR, DATA_DIR


def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr["smiles"]
            vocab.add(attr["label"])
            for i, s in attr["inter_label"]:
                vocab.add((smiles, s))
    return vocab


def get_vocab(dataset_name: str, num_processes: int):
    smiles_path = SMILES_DIR / dataset_name / "all.txt"
    with open(smiles_path, "r") as file:
        data = file.readlines()

    batch_size = len(data) // num_processes + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(num_processes)
    vocab_list = pool.map(process, batches)
    vocab = [(x, y) for vocab in vocab_list for x, y in vocab]
    vocab = list(set(vocab))

    # write lines to output file
    vocab_path = DATA_DIR / "HIERVAE" / dataset_name / "hiervae_vocab.txt"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        for x, y in sorted(vocab):
            print(x, y, file=f)
