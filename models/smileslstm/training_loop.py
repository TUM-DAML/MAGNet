import argparse
import json
import os
import time
from argparse import Namespace
from pathlib import Path

import torch
from guacamol.utils.helpers import setup_default_logger

from models.global_utils import get_model_config, SMILES_DIR, WB_LOG_DIR
from models.smileslstm.src.smiles_rnn_distribution_learner import (
    SmilesRnnDistributionLearner,
)


def run_smiles_training(seed, dataset):
    args = get_model_config("smileslstm", dataset)
    args = Namespace(**args)
    setup_default_logger()
    torch.manual_seed(seed)

    train_data = SMILES_DIR / args.train_data
    valid_data = SMILES_DIR / args.valid_data
    output_dir = WB_LOG_DIR / "SMILES-LSTM" / dataset / str(time.time())
    output_dir.mkdir(parents=True)

    trainer = SmilesRnnDistributionLearner(
        output_dir=output_dir,
        n_epochs=args.n_epochs,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_len=args.max_len,
        batch_size=args.batch_size,
        rnn_dropout=args.rnn_dropout,
        lr=args.lr,
        valid_every=args.valid_every,
    )

    training_set_file = train_data
    validation_set_file = valid_data

    with open(training_set_file) as f:
        train_list = f.readlines()

    with open(validation_set_file) as f:
        valid_list = f.readlines()

    trainer.train(training_set=train_list, validation_set=valid_list)

    print("All done, your trained model is in {args.output_dir}")
