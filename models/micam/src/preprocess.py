import json
import os.path as path
import sys
from argparse import Namespace
from pathlib import Path

import torch
import yaml
from arguments import parse_arguments

from models.global_utils import SOURCE_DIR
from models.micam.src.make_training_data import make_trainig_data
from models.micam.src.merging_operation_learning import merging_operation_learning
from models.micam.src.model.mol_graph import MolGraph
from models.micam.src.model.mydataclass import Paths
from models.micam.src.motif_vocab_construction import motif_vocab_construction

if __name__ == "__main__":
    dataset = "zinc"
    config = SOURCE_DIR / "config" / "paths.yaml"
    config = yaml.safe_load(open(config, "r"))
    args = json.load(open("model_config.json", "r"))[dataset]
    args = Namespace(**args)
    args.dataset = dataset
    args.job_name = "preproc"
    paths = Paths(args, config)

    if not path.exists(paths.operation_path):
        learning_trace = merging_operation_learning(
            train_path=paths.train_path,
            operation_path=paths.operation_path,
            num_iters=args.num_iters,
            min_frequency=args.min_frequency,
            num_workers=args.num_workers,
            mp_threshold=args.mp_thd,
        )

    MolGraph.load_operations(paths.operation_path, args.num_operations)

    if not path.exists(paths.vocab_path):
        mols, vocab = motif_vocab_construction(
            train_path=paths.train_path,
            vocab_path=paths.vocab_path,
            operation_path=paths.operation_path,
            num_operations=args.num_operations,
            mols_pkl_dir=paths.mols_pkl_dir,
            num_workers=args.num_workers,
        )

    MolGraph.load_vocab(paths.vocab_path)

    torch.multiprocessing.set_sharing_strategy("file_system")
    make_trainig_data(
        mols_pkl_dir=paths.mols_pkl_dir,
        valid_path=paths.valid_path,
        vocab_path=paths.vocab_path,
        train_processed_dir=paths.train_processed_dir,
        valid_processed_dir=paths.valid_processed_dir,
        vocab_processed_path=paths.vocab_processed_path,
        num_workers=args.num_workers,
    )
