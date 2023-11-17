import json
import logging
import os
import os.path as path
import random
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from models.global_utils import get_model_config
from models.micam.src.model.dataset import MolsDataset, batch_collate
from models.micam.src.model.MiCaM_VAE import MiCaM, VAE_Output
from models.micam.src.model.mol_graph import MolGraph
from models.micam.src.model.mydataclass import ModelParams, Paths, TrainingParams
from models.micam.src.model.scheduler import beta_annealing_schedule


def train(seed, dataset, config):
    torch.manual_seed(seed)
    random.seed(seed)

    args = get_model_config(config, "micam", dataset)
    args = Namespace(**args)
    args.dataset = dataset
    args.job_name = "training"
    paths = Paths(args, config)
    paths.model_save_dir = Path(config["BASELINE_DIR"]) / "wb_logs" / "MICAM" / dataset / str(time.time())

    model_params = ModelParams(args, config)
    training_params = TrainingParams(args)

    MolGraph.load_operations(paths.operation_path)
    MolGraph.load_vocab(paths.vocab_path)

    os.makedirs(paths.output_dir)
    log_file = path.join(paths.output_dir, "train.log")
    print(f"See {log_file} for log.")
    logging.basicConfig(filename=log_file, filemode="w", format="[%(asctime)s]: %(message)s", level=logging.INFO)

    model = MiCaM(model_params).cuda()
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)

    total_step, beta = 0, training_params.beta_min

    logging.info("HyperParameters:")
    logging.info(model_params)
    logging.info(training_params)

    scheduler = lr_scheduler.ExponentialLR(optimizer, training_params.lr_anneal_rate)
    beta_scheduler = beta_annealing_schedule(params=training_params, init_beta=beta, init_step=total_step)
    train_dataset = MolsDataset(paths.train_processed_dir)

    logging.info(f"Begin training...")
    os.makedirs(paths.model_save_dir)
    stop_train = False
    while True:
        for input in DataLoader(
            dataset=train_dataset, batch_size=training_params.batch_size, shuffle=True, collate_fn=batch_collate
        ):
            total_step += 1
            model.zero_grad()

            input = input.cuda()
            output: VAE_Output = model(input, beta=beta, prop_weight=training_params.prop_weight)

            output.total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_params.grad_clip_norm)

            optimizer.step()
            # output.log_tb_results(total_step, beta, scheduler.get_last_lr()[0])

            if total_step % 50 == 0:
                output.print_results(total_step, lr=scheduler.get_last_lr()[0], beta=beta)

            if total_step % training_params.lr_anneal_iter == 0:
                scheduler.step()

            beta = beta_scheduler.step()

            if total_step == training_params.steps:
                stop_train = True
                break
        model.eval()
        model.zero_grad()
        torch.cuda.empty_cache()
        model_path = path.join(paths.model_save_dir, "model.ckpt")
        motifs_embed_path = path.join(paths.model_save_dir, "motifs_embed.ckpt")
        with torch.no_grad():
            ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
            torch.save(ckpt, model_path)
            model.save_motifs_embed(motifs_embed_path)

        if stop_train:
            break
