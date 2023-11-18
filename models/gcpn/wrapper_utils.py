import json
import time
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem
from torch import optim
from torchdrug import core, data, models, tasks
from torchdrug.core import Engine
from torchdrug.core import Registry as R

from models.global_utils import get_model_config, smiles_from_file
from models.inference import InferenceBase


@R.register("datasets.Solvation")
class mymoldataset(data.MoleculeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_all_modules(config: dict, dataset_name: str):
    BASELINE_DIR = Path(config["BASELINE_DIR"])
    args = get_model_config(config, "gcpn", dataset_name)
    dataset = mymoldataset()
    input_smiles = smiles_from_file(BASELINE_DIR / "smiles_files" / dataset_name / "train.txt")
    dataset.load_smiles(smiles_list=input_smiles, targets=dict(), kekulize=True, atom_feature="symbol")
    model = models.RGCN(
        input_dim=dataset.node_feature_dim,
        num_relation=dataset.num_bond_type,
        hidden_dims=args["hidden_dims"],
        batch_norm=args["batch_norm"],
    )
    task = tasks.GCPNGeneration(
        model,
        dataset.atom_types,
        max_edge_unroll=args["max_edge_unroll"],
        max_node=args["max_node"],
        criterion=args["criterion"],
    )
    optimizer = optim.Adam(task.parameters(), lr=args["lr"])
    solver = core.Engine(
        task, dataset, None, None, optimizer, gpus=(0,), batch_size=args["batch_size"], log_interval=500
    )
    return solver, task, model, dataset, args


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> Engine:
    solver, task, _, _, _ = get_all_modules(config, dataset)
    solver.load(Path(config["BASELINE_DIR"]) / "model_ckpts" / "GCPN" / dataset / (model_id + ".pkl"))
    return InferenceGCPN(model=task, config=config, seed=seed)


def run_training(seed: int, dataset: str, config: dict):
    model_name = str(time.time())
    solver, _, _, _, args = get_all_modules(config, dataset)
    solver.train(num_epoch=args["num_epochs"])
    solver.save(Path(config["BASELINE_DIR"]) / "wb_logs" / "GCPN" / dataset / (model_name + ".pkl"))


class InferenceGCPN(InferenceBase):
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
