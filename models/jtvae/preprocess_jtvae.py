import pickle
from multiprocessing import Pool
from optparse import OptionParser

import rdkit
import tqdm

from models.jtvae.fastjt.mol_tree import MolTree


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


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-o", "--out", dest="out_dir")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(opts.train_path) as f:
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

        with open(opts.out_dir + "tensors-%d.pkl" % split_id, "wb") as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
