import os
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch

from models.hiervae.src.hgnn import HierVAE
from models.hiervae.src.mol_graph import MolGraph
from models.hiervae.src.vocab import PairVocab, common_atom_vocab
from models.hiervae.training_loop import run_hiervae_training
from models.hiervae.preprocess_hiervae import preprocess_hiervae
from models.hiervae.get_vocab import get_vocab
from models.inference import InferenceBase
from models.global_utils import BASELINE_DIR, CKPT_DIR, WB_CONFIG


def get_model_func(dataset: str, model_id: str, seed: int) -> HierVAE:
    path = CKPT_DIR / WB_CONFIG["WB_PROJECT"] / model_id / "checkpoints"
    checkpoint = max(os.listdir(path), key=lambda x: int(x[6:].split("-")[0]))
    print("Using HIERVAE checkpoint: ", checkpoint)
    path = path / checkpoint
    vocab = PairVocab(dataset, BASELINE_DIR)
    model = HierVAE(vocab=vocab)
    model.load_state_dict(torch.load(path)["state_dict"])
    model.cuda()
    return InferenceHIERVAE(model=model, seed=seed)


def run_training(seed: int, dataset: str):
    run_hiervae_training(seed, dataset)

def run_preprocessing(dataset_name: str, num_processes: int):
    print("Running HIERVAE preprocessing...")
    get_vocab(dataset_name, num_processes)
    preprocess_hiervae(dataset_name, num_processes)


class InferenceHIERVAE(InferenceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def random_mol(self, num_samples):
        return self.model.sample(batch_size=num_samples, greedy=False)

    def encode(self, smiles):
        _, tensors, _ = tensorize(smiles, self.model.encoder.vocab)
        tree_tensors, graph_tensors = make_cuda(tensors)
        root_vecs, _, _, _ = self.model.encoder(tree_tensors, graph_tensors)
        root_vecs, _ = self.model.rsample(root_vecs, self.model.R_mean, self.model.R_var, perturb=False)
        return root_vecs.detach().cpu().numpy()

    def decode(self, latent):
        root_vecs = torch.tensor(latent).cuda()
        return self.model.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).long().cuda() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).long().cuda() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c
