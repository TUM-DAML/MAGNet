import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import rdkit
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from models.global_utils import get_model_config, BASELINE_DIR, WB_CONFIG, WB_LOG_DIR
from models.jtvae.fastjt.datautils import MolTreeFolder
from models.jtvae.fastjt.jtnn_vae import JTNNVAE

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def run_jtvae_training(seed, dataset):
    kwargs = get_model_config("jtvae", dataset)
    torch.manual_seed(seed=seed)
    train_loader = MolTreeFolder(dataset)
    model = JTNNVAE(
        vocab=train_loader.vocab,
        beta=kwargs["beta"],
        kl_anneal=kwargs["kl_anneal"],
        kl_warmup=kwargs["kl_warmup"],
        step_beta=kwargs["step_beta"],
        lr=kwargs["lr"],
    ).cuda()
    logger = pl.loggers.WandbLogger(
        entity=WB_CONFIG["WB_ENTITY"],
        project=WB_CONFIG["WB_PROJECT"],
        save_dir=str(WB_LOG_DIR),
        name="JTVAE",
    )
    wandb.init()
    wandb.config.update(kwargs)
    checkpointing = ModelCheckpoint(save_top_k=-1, save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        log_every_n_steps=kwargs["log_every"],
        max_epochs=kwargs["epochs"],
        devices=1,
        check_val_every_n_epoch=100,
        gradient_clip_val=kwargs["clip_norm"],
        callbacks=[checkpointing],
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, None)
    # return WANDB run id
    return Path(trainer.logger.experiment.dir).parts[-2]
