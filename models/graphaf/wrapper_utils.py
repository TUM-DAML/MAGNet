import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem
from torch import nn, optim
from torchdrug import core, data, models, tasks
from torchdrug.core import Engine
from torchdrug.core import Registry as R
from torchdrug.layers import distribution

from baselines.global_utils import smiles_from_file, get_model_config
from baselines.inference import InferenceBase


@R.register("datasets.Solvation")
class mymoldataset(data.MoleculeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_all_modules(config: dict, dataset_name: str):
    BASELINE_DIR = Path(config["BASELINE_DIR"])
    args = get_model_config(config, "graphaf", dataset_name)
    dataset = mymoldataset()
    input_smiles = smiles_from_file(BASELINE_DIR / "smiles_files" / dataset_name / "train.txt")
    dataset.load_smiles(smiles_list=input_smiles, targets=dict(), kekulize=True, atom_feature="symbol")
    model = models.RGCN(
        input_dim=dataset.num_atom_type,
        num_relation=dataset.num_bond_type,
        hidden_dims=args["hidden_dims"],
        batch_norm=args["batch_norm"],
    )

    num_atom_type = dataset.num_atom_type
    # add one class for non-edge
    num_bond_type = dataset.num_bond_type + 1

    node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type), torch.ones(num_atom_type))
    edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type), torch.ones(num_bond_type))
    node_flow = models.GraphAF(model, node_prior, num_layer=args["node_num_layers"])
    edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=args["edge_num_layers"])

    task = tasks.AutoregressiveGeneration(
        node_flow,
        edge_flow,
        max_node=args["max_node"],
        max_edge_unroll=args["max_edge_unroll"],
        criterion=args["criterion"],
    )
    optimizer = optim.Adam(task.parameters(), lr=args["lr"])
    solver = core.Engine(
        task, dataset, None, None, optimizer, gpus=(0,), batch_size=args["batch_size"], log_interval=500
    )
    return solver, task, model, dataset, args


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> Engine:
    solver, task, _, _, _ = get_all_modules(config, dataset)
    solver.load(Path(config["BASELINE_DIR"]) / "model_ckpts" / "GRAPHAF" / dataset / (model_id + ".pkl"))
    return InferenceGRAPHAF(model=task, config=config, seed=seed)


def run_training(seed: int, dataset: str, config: dict):
    model_name = str(time.time())
    solver, _, _, _, args = get_all_modules(config, dataset)
    solver.train(num_epoch=args["num_epochs"])
    solver.save(Path(config["BASELINE_DIR"]) / "wb_logs" / "GRAPHAF" / dataset / (model_name + ".pkl"))


class InferenceGRAPHAF(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        return self.model.generate(num_sample=num_samples, max_resample=2).to_smiles()

    def encode(self, smiles):
        raise NotImplementedError()

    def decode(self, latent):
        raise NotImplementedError()

    def valid_check(self, smiles):
        return Chem.MolFromSmiles(smiles).GetNumAtoms() > 2
