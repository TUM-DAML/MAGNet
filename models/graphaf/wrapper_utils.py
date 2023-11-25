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

from models.global_utils import get_model_config, smiles_from_file, SMILES_DIR, CKPT_DIR, WB_LOG_DIR
from models.inference import InferenceBase


@R.register("datasets.Solvation")
class mymoldataset(data.MoleculeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_all_modules(dataset_name: str):
    args = get_model_config("graphaf", dataset_name)
    dataset = mymoldataset()
    input_smiles = smiles_from_file(SMILES_DIR / dataset_name / "train.txt")
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


def get_model_func(dataset: str, model_id: str, seed: int) -> Engine:
    solver, task, _, _, _ = get_all_modules(dataset)
    solver.load(CKPT_DIR / "GRAPHAF" / dataset / (model_id + ".pkl"))
    return InferenceGRAPHAF(model=task, seed=seed)


def run_training(seed: int, dataset: str):
    model_name = str(time.time())
    solver, _, _, _, args = get_all_modules(dataset)
    solver.train(num_epoch=args["num_epochs"])
    save_path = WB_LOG_DIR / "GRAPHAF" / dataset
    save_path.mkdir(parents=True, exist_ok=True)
    solver.save(save_path / (model_name + ".pkl"))


def run_preprocessing(dataset_name: str, num_processes: int):
    print("GRAPHAF does not need preprocessing, terminating now...")


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
        return (Chem.MolFromSmiles(smiles).GetNumAtoms() > 2) and (Chem.MolFromSmiles(smiles) is not None)
