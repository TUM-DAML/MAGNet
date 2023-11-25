from pathlib import Path
from typing import List

from rdkit import Chem

from models.inference import InferenceBase
from models.global_utils import CKPT_DIR
from models.smileslstm.src.rnn_model import SmilesRnn
from models.smileslstm.src.rnn_utils import load_rnn_model
from models.smileslstm.src.smiles_rnn_generator import SmilesRnnGenerator
from models.smileslstm.training_loop import run_smiles_training


def get_model_func(dataset: str, model_id: str, seed: int) -> SmilesRnn:
    model_path = CKPT_DIR / "SMILES-LSTM" / dataset / str(model_id) / "model_final.pt"
    model_def = Path(model_path).with_suffix(".json")
    model = load_rnn_model(model_def, model_path, "cuda", copy_to_cpu=True)
    return InferenceSMILESLSTM(model=model, seed=seed)


def run_training(seed: int, dataset: str):
    run_smiles_training(seed, dataset)

def run_preprocessing(dataset_name: str, num_processes: int):
    print("SMILES-LSTM does not need preprocessing, terminating now...")


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
