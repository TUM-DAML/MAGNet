import json
from pathlib import Path

import pytorch_lightning as pl
import rdkit
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from models.global_utils import get_model_config, BASELINE_DIR, WB_CONFIG, WB_LOG_DIR
from models.hiervae.src.dataset import DataFolder
from models.hiervae.src.hgnn import HierVAE
from models.hiervae.src.vocab import PairVocab

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def run_hiervae_training(seed, dataset):
    torch.manual_seed(seed)
    kwargs = get_model_config("hiervae", dataset)
    vocab = PairVocab(dataset, BASELINE_DIR)
    train_loader = DataFolder(dataset, kwargs["batch_size"], BASELINE_DIR)
    model = HierVAE(
        vocab,
        beta=kwargs["beta"],
        kl_anneal=kwargs["kl_anneal"],
        kl_warmup=kwargs["kl_warmup"],
        step_beta=kwargs["step_beta"],
    )
    logger = pl.loggers.WandbLogger(
        entity=WB_CONFIG["WB_ENTITY"],
        project=WB_CONFIG["WB_PROJECT"],
        save_dir=str(WB_LOG_DIR),
        name="HierVAE",
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
        enable_progress_bar=kwargs["progress_bar"],
    )
    trainer.fit(model, train_loader, None)
    return Path(trainer.logger.experiment.dir).parts[-2]
