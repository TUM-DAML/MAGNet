from pathlib import Path
from typing import List

from rdkit import Chem

from models.inference import InferenceBase
from models.smileslstm.src.rnn_model import SmilesRnn
from models.smileslstm.src.rnn_utils import load_rnn_model
from models.smileslstm.src.smiles_rnn_generator import SmilesRnnGenerator
from models.smileslstm.training_loop import run_smiles_training


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> SmilesRnn:
    BASELINE_DIR = Path(config["BASELINE_DIR"])
    model_path = BASELINE_DIR / "model_ckpts" / "SMILES-LSTM" / dataset / str(model_id) / "model_final.pt"
    model_def = Path(model_path).with_suffix(".json")
    model = load_rnn_model(model_def, model_path, "cuda", copy_to_cpu=True)
    return InferenceSMILESLSTM(model=model, config=config, seed=seed)


def run_training(seed: int, dataset: str, config: dict):
    run_smiles_training(seed, dataset, config)


class InferenceSMILESLSTM(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        generator = SmilesRnnGenerator(model=self.model, device="cuda")
        output_smiles = generator.generate(num_samples)
        return output_smiles

    def encode(self, smiles):
        raise NotImplementedError()

    def decode(self, latent):
        raise NotImplementedError()

    def valid_check(self, smiles):
        return Chem.MolFromSmiles(smiles) is not None
