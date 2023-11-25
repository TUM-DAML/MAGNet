import os
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from rdkit import Chem

from models.global_utils import get_model_config, CKPT_DIR, WB_CONFIG, SMILES_DIR, DATA_DIR
from models.inference import InferenceBase
from models.magnet.src.utils import manual_batch_to_device
from models.psvae.data.bpe_dataset import get_dataloader
from models.psvae.data.mol_bpe import Tokenizer, graph_bpe
from models.psvae.pl_models.ps_vae_model import PSVAEModel
from models.psvae.training_loop import train


def get_model_func(dataset: str, model_id: str, seed: int) -> PSVAEModel:
    model_path = CKPT_DIR / WB_CONFIG["WB_PROJECT"] / model_id / "checkpoints"
    checkpoint = sorted(os.listdir(model_path))[-1]
    print("Using PSVAE checkpoint: ", checkpoint)
    model_path = model_path / checkpoint
    model = PSVAEModel.load_from_checkpoint(model_path)
    model.eval()
    model.cuda()
    model.dataset_name = dataset
    return InferencePSVAE(model=model, seed=seed)


def run_training(seed: int, dataset: str):
    train(seed, dataset)

def run_preprocessing(dataset_name: str, num_processes: int):
    print("Running PSVAE preprocessing...")
    print("Extracting vocabulary...")
    vocab_size = get_model_config("psvae", dataset_name)["vocab_size"]
    vocab_path = DATA_DIR / "PSVAE" / dataset_name
    vocab_path.mkdir(parents=True, exist_ok=True)
    data_path = SMILES_DIR / dataset_name / "all.txt"
    graph_bpe(data_path, vocab_len=vocab_size, vocab_path=vocab_path / f"psvae_{vocab_size}.txt", cpus=num_processes)
    print("PSVAE does file preprocessing on the fly, so we are done here!")


class InferencePSVAE(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        zs = self.model.sample_z(num_samples, "cuda")
        sampled_smiles = []
        for z in zs:
            mol = self.model.inference_single_z(z, max_atom_num=60, add_edge_th=0.5, temperature=0.8)
            sampled_smiles.append(Chem.MolToSmiles(mol))
        return sampled_smiles

    def encode(self, smiles):
        train_dataloader = prepare_data_loader(
            smiles, self.model.dataset_name, batch_size=len(smiles)
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


def prepare_data_loader(smiles, dataset_name, batch_size=1):
    vocab_size = get_model_config("psvae", dataset_name)["vocab_size"]
    vocab_path = DATA_DIR / "PSVAE" / dataset_name / f"psvae_{vocab_size}.txt"
    tokenizer = Tokenizer(vocab_path)
    # this train path is just a dummy anyway
    train_path = SMILES_DIR / "zinc" / "train.txt"
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
