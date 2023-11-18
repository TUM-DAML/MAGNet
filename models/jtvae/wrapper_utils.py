import os
from pathlib import Path

import torch

from models.inference import InferenceBase
from models.jtvae.fastjt.datautils import MolTreeFolder
from models.jtvae.fastjt.jtnn_vae import JTNNVAE
from models.jtvae.training_loop import run_jtvae_training


def run_training(seed: int, dataset: str, config: dict):
    run_jtvae_training(seed, dataset, config)


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> InferenceBase:
    baseline_dir = Path(config["BASELINE_DIR"])
    path = baseline_dir / "model_ckpts" / "JTVAE" / dataset / model_id / "checkpoints"
    checkpoint = sorted(os.listdir(path))[-1]
    print("Using JTVAE checkpoint: ", checkpoint)
    train_loader = MolTreeFolder(dataset, baseline_dir, batch_size=16, partition="train", shuffle=False)
    model = JTNNVAE(vocab=train_loader.vocab)
    model.load_state_dict(torch.load(path / checkpoint)["state_dict"])
    model.cuda()
    return InferenceJTVAE(model=model, config=config, seed=seed)


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
