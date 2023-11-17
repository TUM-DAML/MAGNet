from typing import List
from pathlib import Path
import numpy as np
import torch

from baselines.magnet.src.model.load_utils import load_model_from_id
from baselines.magnet.src.model.flow_vae import FlowMAGNet
from baselines.magnet.src.model.vae import MAGNet
from baselines.magnet.train import run_magnet_latent_train, run_magnet_vae_training
from baselines.inference import InferenceBase


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> FlowMAGNet:
    model = load_model_from_id(
        data_dir=Path(config["BASELINE_DIR"]),
        collection=config["WB_PROJECT"],
        run_id=model_id,
        model_class=FlowMAGNet,
        model_class=MAGNet,
        seed_model=seed,
    )
    return InferenceMAGNET(model=model, config=config, seed=seed)


def run_training(seed: int, dataset: str, config: dict):
    magnet_id = run_magnet_vae_training(seed, dataset, config)
    run_magnet_latent_train(seed, dataset, config, magnet_id)


class InferenceMAGNET(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        return self.model.sample_molecules(num_samples, largest_comp=False)

    def encode(self, smiles):
        batch = self.model.batch_from_smiles(smiles)
        latents = self.model.encode_to_latent_mean(batch)
        return latents.detach().cpu().numpy()

    def decode(self, latent):
        embedding = torch.tensor(latent).to(self.model.device)
        return self.model.decode_from_latent_mean(embedding)

    def valid_check(self, smiles):
        return "." not in smiles
