import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from models.global_utils import get_model_config, SMILES_DIR, DATA_DIR, WB_CONFIG, WB_LOG_DIR
from models.psvae.data import bpe_dataset
from models.psvae.data.mol_bpe import Tokenizer
from models.psvae.pl_models import PSVAEModel
from models.psvae.utils.logger import print_log
from models.psvae.utils.nn_utils import (
    VAEEarlyStopping,
    common_config,
    encoder_config,
    predictor_config,
    ps_vae_config,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(seed, dataset):
    args = get_model_config("psvae", dataset)
    args = Namespace(**args)
    setup_seed(seed)
    args.vocab = DATA_DIR / "PSVAE" / args.vocab
    args.train_set = SMILES_DIR / args.train_set
    args.valid_set = SMILES_DIR / args.valid_set
    args.test_set = SMILES_DIR / args.test_set

    tokenizer = Tokenizer(args.vocab)
    vocab = tokenizer.chem_vocab
    train_loader = bpe_dataset.get_dataloader(
        args.train_set,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    valid_loader = bpe_dataset.get_dataloader(
        args.valid_set,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    # config and create model
    print("creating model ...")
    config = {
        **common_config(args),
        **encoder_config(args, vocab),
        **predictor_config(args),
    }
    # config of encoder is also updated
    wandb.init()
    config.update(ps_vae_config(args, tokenizer))
    model = PSVAEModel(config, tokenizer)
    print_log(f"config: {config}")
    print(model)

    # train from start
    print("start training")
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, every_n_epochs=1)
    logger = pl.loggers.WandbLogger(
        entity=WB_CONFIG["WB_ENTITY"],
        project=WB_CONFIG["WB_PROJECT"],
        save_dir=str(WB_LOG_DIR),
        name="PSVAE",
    )
    config.update(vars(args))
    wandb.config.update(config)
    trainer_config = {
        "devices": 1,
        "max_epochs": args.epochs,
        "callbacks": [checkpoint_callback],
        "gradient_clip_val": args.grad_clip,
        "logger": logger,
        "log_every_n_steps": 1,
    }
    if len(args.gpus.split(",")) > 1:
        trainer_config["accelerator"] = "dp"
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, train_loader, valid_loader)
