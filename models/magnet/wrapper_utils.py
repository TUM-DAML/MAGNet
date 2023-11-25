from typing import List
from pathlib import Path
import numpy as np
import torch

from models.magnet.src.model.load_utils import load_model_from_id
from models.magnet.src.model.flow_vae import FlowMAGNet
from models.magnet.src.model.vae import MAGNet
from models.magnet.train import run_magnet_latent_train, run_magnet_vae_training
from models.inference import InferenceBase
from models.global_utils import WB_CONFIG, BASELINE_DIR
from models.magnet.preprocessing import run_magnet_preproc


def get_model_func(dataset: str, model_id: str, seed: int) -> FlowMAGNet:
    model = load_model_from_id(
        collection=WB_CONFIG["WB_PROJECT"],
        run_id=model_id,
        model_class=FlowMAGNet,
        seed_model=seed,
    )
    return InferenceMAGNET(model=model, seed=seed)


def run_training(seed: int, dataset: str):
    magnet_id = run_magnet_vae_training(seed, dataset)
    run_magnet_latent_train(seed, dataset, magnet_id)

def run_preprocessing(dataset_name: str, num_processes: int):
    print("Running MAGNet preprocessing...")
    run_magnet_preproc(dataset_name, num_processes)


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
