import os
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from rdkit import Chem

from baselines.inference import InferenceBase
from baselines.psvae.data.bpe_dataset import get_dataloader
from baselines.psvae.data.mol_bpe import Tokenizer
from baselines.psvae.pl_models.ps_vae_model import PSVAEModel
from baselines.psvae.training_loop import train
from baselines.magnet.src.utils import manual_batch_to_device


def get_model_func(dataset: str, model_id: str, seed: int, config: dict) -> PSVAEModel:
    model_path = Path(config["BASELINE_DIR"]) / "model_ckpts" / "PSVAE" / dataset / model_id / "checkpoints"
    checkpoint = sorted(os.listdir(model_path))[-1]
    print("Using PSVAE checkpoint: ", checkpoint)
    model_path = model_path / checkpoint
    model = PSVAEModel.load_from_checkpoint(model_path)
    model.eval()
    model.cuda()
    model.dataset_name = dataset
    return InferencePSVAE(model=model, config=config, seed=seed)


def run_training(seed: int, dataset: str, config: dict):
    train(seed, dataset, config)


class InferencePSVAE(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        zs = self.model.sample_z(num_samples, "cuda")
        sampled_smiles = []
        for z in tqdm.tqdm(zs):
            mol = self.model.inference_single_z(z, max_atom_num=60, add_edge_th=0.5, temperature=0.8)
            sampled_smiles.append(Chem.MolToSmiles(mol))
        return sampled_smiles

    def encode(self, smiles):
        train_dataloader = prepare_data_loader(
            smiles, self.model.dataset_name, Path(self.config["BASELINE_DIR"]), batch_size=len(smiles)
        )
        batch = next(iter(train_dataloader))
        manual_batch_to_device(batch, "cuda")
        z_graph = get_embedding(self.model, batch, sample=False)
        return z_graph.detach().cpu().numpy()

    def decode(self, latent):
        decoded_smiles = []
        for l in latent:
            gen_mol = self.model.decoder.inference(
                z=torch.Tensor(l).cuda(),
                max_atom_num=60,
                add_edge_th=0.5,
                temperature=0.8,
            )
            decoded_smiles.append(Chem.MolToSmiles(gen_mol))
        return decoded_smiles


def prepare_data_loader(smiles, dataset_name, BASELINE_DIR, batch_size=1):
    tokenizer = Tokenizer(BASELINE_DIR / "data" / "PSVAE" / dataset_name / "psvae_512.txt")
    # this train path is just a dummy anyway
    train_path = BASELINE_DIR / "smiles_files" / "zinc" / "train.txt"
    train_loader = get_dataloader(
        train_path,
        tokenizer,
        batch_size=batch_size,
        smiles=smiles,
        shuffle=False,
        num_workers=1,
        disable_tqdm=True,
    )
    return train_loader


def get_embedding(model, batch, sample):
    x, edge_index, edge_attr = batch["x"], batch["edge_index"], batch["edge_attr"]
    x_pieces, x_pos = batch["x_pieces"], batch["x_pos"]
    x = model.decoder.embed_atom(x, x_pieces, x_pos)
    batch_size, node_num, node_dim = x.shape
    graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)
    _, all_x = model.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
    # [batch_size, dim_graph_feature]
    graph_embedding = model.encoder.embed_graph(all_x, graph_ids, batch["atom_mask"].flatten())
    if not sample:
        mean_embedding = model.decoder.W_mean(graph_embedding)
        return mean_embedding
    sampled_emb, _ = model.decoder.rsample(graph_embedding)
    return sampled_emb
