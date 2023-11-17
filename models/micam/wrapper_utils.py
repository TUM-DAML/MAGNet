import json
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np

from baselines.global_utils import get_model_config
from baselines.micam.src.model.MiCaM_VAE import MiCaM
from baselines.micam.src.model.mydataclass import ModelParams, Paths
from baselines.micam.src.train import train


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> MiCaM:
    args = get_model_config(config, "micam", dataset)
    args = Namespace(**args)
    args.dataset = dataset
    args.job_name = "MICAM"
    paths = Paths(args, config)
    paths.model_dir = Path(config["BASELINE_DIR"]) / "model_ckpts" / "MICAM" / dataset / model_id
    model_params = ModelParams(args, config)
    generator = MiCaM.load_generator(model_params, paths)
    return generator


def sample_func(model: MiCaM, num_samples: int, config: dict):
    output_smiles = model.generate(num_samples)
    return output_smiles


def encode_func(model: MiCaM, smiles: List[str], config: dict) -> List[np.array]:
    return []


def decode_func(model: MiCaM, embeddings: List[np.array], config: dict) -> List[str]:
    return []


def run_training(seed: int, dataset: str, config: dict):
    return train(seed, dataset, config)
