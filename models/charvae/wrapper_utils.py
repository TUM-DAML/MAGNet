from typing import List

import numpy as np
from rdkit import Chem

from models.charvae.src.vae_utils import VAEUtils
from models.charvae.train_vae import main_no_prop
from models.global_utils import get_model_config
from models.inference import InferenceBase


def get_model_func(dataset: str, model_id: str, seed: int) -> VAEUtils:
    model_config = get_model_config("charvae", dataset)
    encoder_file = model_config["encoder_weights_file"]
    decoder_file = model_config["decoder_weights_file"]

    model = VAEUtils(
        encoder_file=encoder_file,
        decoder_file=decoder_file,
        dataset=dataset,
        seed=seed,
        id=model_id,
    )
    return InferenceCHARVAE(model=model, seed=seed)


def run_preprocessing(dataset_name: str, num_processes: int):
    print("CHARVAE does not need preprocessing, terminating now...")


class InferenceCHARVAE(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        z_1 = np.random.normal(size=(num_samples, self.model.mu.shape[-1]))
        X_r = self.model.decode(z_1)
        output_smiles = self.model.hot_to_smiles(X_r, strip=True)
        return output_smiles

    def encode(self, smiles):
        raise NotImplementedError()

    def decode(self, latent):
        raise NotImplementedError()

    def valid_check(self, smiles):
        return Chem.MolFromSmiles(smiles) is not None


def run_training(seed: int, dataset: str):
    main_no_prop(seed, dataset)
