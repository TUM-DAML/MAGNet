import argparse
import bz2
from pathlib import PosixPath

import _pickle as cPickle
import tqdm
from torch.multiprocessing import Pool

from baselines.magnet.src.chemutils.hypergraph import MolDecomposition
from baselines.magnet.src.utils import smiles_from_file


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--num_processes", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    args.path = PosixPath(args.path)

    # define output folder
    output_dir = args.path.parent / (args.output_prefix + args.path.stem)
    assert not output_dir.is_dir()
    output_dir.mkdir(exist_ok=True)
    # check whether output_dir exists as a directory

    all_smiles = smiles_from_file(args.path)
    all_smiles = [(output_dir, i, sm) for (i, sm) in enumerate(all_smiles)]

    def process_func(input):
        output_dir, i, smiles = input
        mol_holder = MolDecomposition(smiles)
        fp = output_dir / f"{i:06d}.pbz2"
        with bz2.BZ2File(fp, "w") as f:
            cPickle.dump((i, smiles, mol_holder), f)
        return None

    # Multiprocessing but with Progress Bar
    with Pool(processes=args.num_processes) as p:
        outputs = []
        max_ = len(all_smiles)
        with tqdm.tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(process_func, all_smiles):
                pbar.update()

    print("Number of Saved Files: ", len(list(output_dir.glob("*.pbz2"))))
