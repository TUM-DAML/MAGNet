import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from rdkit import RDLogger

from models.global_utils import get_model_config, BASELINE_DIR, WB_CONFIG, WB_LOG_DIR
from models.magnet.src.data.latent_module import LatentDataModule
from models.magnet.src.data.mol_module import MolDataModule
from models.magnet.src.model.flow_vae import FlowMAGNet, get_flow_training_args
from models.magnet.src.model.load_utils import load_model_from_id
from models.magnet.src.model.vae import MAGNet
from models.magnet.src.utils import save_model_config_to_file

RDLogger.DisableLog("rdApp.*")


def run_magnet_latent_train(seed: int, dataset: str,magnet_id: str):
    kwargs = get_model_config("magnet", dataset)["flow_training"]
    kwargs = get_flow_training_args(kwargs)
    dm = LatentDataModule(
        dataset=dataset,
        collection=WB_CONFIG["WB_PROJECT"],
        magnet_id=magnet_id,
        batch_size=kwargs["batch_size"],
        ndatapoints=kwargs["n_datapoints"],
        num_workers=kwargs["num_workers"],
    )

    # Load VAE model for inference
    model, model_config = load_model_from_id(
        collection=WB_CONFIG["WB_PROJECT"],
        run_id=magnet_id,
        load_config=dict(
            patience=max(1, kwargs["epochs"] // 400),
            lr=kwargs["lr"],
            lr_sch_decay=kwargs["lr_sch_decay"],
            flow_dim_config=kwargs["flow_dim_config"],
            sample_config=kwargs["sample_config"],
            dataset=dataset
        ),
        dataset=dataset,
        model_class=FlowMAGNet,
        seed_model=seed,
        return_config=True,
    )
    model.cuda()
    wandb.init()
    logger = pl.loggers.WandbLogger(
        entity=WB_CONFIG["WB_ENTITY"],
        project=WB_CONFIG["WB_PROJECT"],
        save_dir=str(WB_LOG_DIR),
    )
    wandb.config.update(kwargs)
    logger.experiment
    save_model_config_to_file(
        WB_LOG_DIR / WB_CONFIG["WB_PROJECT"], str(logger.version), model_config, model
    )

    # Train Flow Matching
    checkpointing = ModelCheckpoint(save_last=True, save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        log_every_n_steps=20,
        max_epochs=kwargs["epochs"],
        gradient_clip_val=kwargs["gradient_clip"],
        logger=logger,
        callbacks=[checkpointing],
        enable_progress_bar=False,
        check_val_every_n_epoch=kwargs["epochs"] + 1,
        limit_val_batches=1,
    )
    trainer.fit(model, datamodule=dm)
    wandb.finish()
    return dict()


def run_magnet_vae_training(seed: int, dataset: str):
    torch.manual_seed(seed)
    kwargs = get_model_config("magnet", dataset)["vae_training"]
    dm_kwargs = dict(
        dataset=kwargs["dataset"],
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
    )

    dm = MolDataModule(**dm_kwargs)
    dm.setup()
    wandb.init()
    logger = pl.loggers.WandbLogger(
        entity=WB_CONFIG["WB_ENTITY"],
        project=WB_CONFIG["WB_PROJECT"],
        save_dir=str(WB_LOG_DIR),
    )
    wandb.config.update(kwargs)

    model_kwargs = dict(
        lr=kwargs["lr"],
        lr_sch_decay=kwargs["lr_sch_decay"],
        dim_config=kwargs["dim_config"],
        layer_config=kwargs["layer_config"],
        loss_weights=kwargs["loss_weights"],
        beta_annealing=kwargs["beta_annealing"],
    )
    model = MAGNet(
        feature_sizes=dm.feature_sizes,
        **model_kwargs,
    ).cuda()

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        max_epochs=kwargs["epochs"],
        check_val_every_n_epoch=1,
        gradient_clip_val=kwargs["gradclip"],
        limit_val_batches=int(50),
    )
    trainer_kwargs.update({"enable_progress_bar": False})
    checkpointing = ModelCheckpoint(monitor="val_loss", filename="model-{epoch:02d}-{val_loss:.2f}", save_last=True)
    trainer_kwargs.update({"logger": logger, "callbacks": [checkpointing]})
    trainer = pl.Trainer(**trainer_kwargs)

    # this needs to be done due to issues with w&b, we cant access the run_id otherwise
    logger.experiment
    save_model_config_to_file(
        WB_LOG_DIR / WB_CONFIG["WB_PROJECT"], str(logger.version), kwargs, model
    )

    trainer.fit(model, datamodule=dm)
    wandb.finish()
    return logger.version
