import os
from pathlib import Path

import torch

from models.inference import InferenceBase
from models.global_utils import CKPT_DIR, SMILES_DIR, DATA_DIR, WB_CONFIG
from models.jtvae.fastjt.datautils import MolTreeFolder
from models.jtvae.fastjt.jtnn_vae import JTNNVAE
from models.jtvae.fastjt.mol_tree import main_mol_tree
from models.jtvae.training_loop import run_jtvae_training
from models.jtvae.preprocess_jtvae import preprocess_jtvae


def run_training(seed: int, dataset: str):
    run_jtvae_training(seed, dataset)


def get_model_func(dataset: str, model_id: str, seed: int) -> InferenceBase:
    path = CKPT_DIR / WB_CONFIG["WB_PROJECT"] / model_id / "checkpoints"
    checkpoint = max(os.listdir(path), key=lambda x: int(x[6:].split("-")[0]))
    print("Using JTVAE checkpoint: ", checkpoint)
    train_loader = MolTreeFolder(dataset, batch_size=16, partition="train", shuffle=False)
    model = JTNNVAE(vocab=train_loader.vocab)
    model.load_state_dict(torch.load(path / checkpoint)["state_dict"])
    model.cuda()
    return InferenceJTVAE(model=model, seed=seed)

def run_preprocessing(dataset_name: str, num_processes: int):
    print("Running JTVAE preprocessing...")
    preprocess_jtvae(dataset_name, num_processes)
    train_path = SMILES_DIR / dataset_name / "all.txt"
    out_dir = DATA_DIR / "JTVAE" / dataset_name / "jtvae_vocab.txt"
    main_mol_tree(train_path, out_dir, MAX_TREE_WIDTH=50)


class InferenceJTVAE(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        return [self.model.sample_prior() for _ in range(num_samples)]

    def encode(self, smiles):
        tree_vec, mol_vec = self.model.encode_from_smiles(smiles)
        tree_mean = self.model.T_mean(tree_vec)
        mol_mean = self.model.G_mean(mol_vec)
        return torch.cat((mol_mean, tree_mean), dim=1).detach().cpu().numpy()

    def decode(self, latent):
        mol_mean, tree_mean = torch.chunk(torch.tensor(latent).cuda(), 2, dim=1)
        outputs = []
        for m, t in zip(mol_mean, tree_mean):
            outputs.append(self.model.decode(m.view(1, -1), t.view(1, -1), prob_decode=False))
        return outputs
