import time
from pathlib import Path
from typing import List

import numpy as np

from baselines.global_utils import get_model_config
from baselines.inference import InferenceBase
from baselines.moler.molecule_generation import load_model_from_directory
from baselines.moler.molecule_generation.cli.train import get_argparser, run_from_args
from baselines.moler.molecule_generation.wrapper import ModelWrapper


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> ModelWrapper:
    model_path = Path(config["BASELINE_DIR"]) / "model_ckpts" / "MOLER" / dataset / model_id
    return InferenceMOLER(model=(load_model_from_directory, model_path), config=config, seed=seed)


def run_training(seed: int, dataset: str, config: dict):
    BASELINE_DIR = Path(config["BASELINE_DIR"])
    kwargs = get_model_config(config, "moler", dataset)
    vocab_size = kwargs["vocab_size"]

    assert dataset in ["zinc", "guacamol"], NotImplementedError()
    data_path = BASELINE_DIR / "data" / "MOLER" / dataset / f"moler_preproc_{vocab_size}" / "trace"

    assert data_path.exists(), ValueError(f"Folder at '{data_path}' does not exists.")

    model = "MoLeR"
    expname = model + "_" + dataset + "_" + str(time.time())
    args = get_argparser().parse_args([model, str(data_path)])
    setattr(args, "run_id", expname)
    setattr(args, "wandb", True)
    setattr(args, "random_seed", seed)
    setattr(args, "max_epochs", 20)
    setattr(args, "save_dir", str(BASELINE_DIR / "wb_logs" / "MOLER" / dataset / expname))
    run_from_args(args)


class InferenceMOLER(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        return [s[0] for s in self.model.sample(num_samples)]

    def encode(self, smiles):
        embeddings = self.model.encode(smiles)
        return np.stack(embeddings)

    def decode(self, latent):
        output = self.model.decode(latent.astype(np.float32))
        decoded_smiles = [d[0] for d in output]
        return decoded_smiles
