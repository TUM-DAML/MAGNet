import os
import pickle
import random
import sys

import torch
from rdkit.Chem import Descriptors, RDConfig
from torch.utils.data import DataLoader, Dataset

from baselines.jtvae.fastjt.jtmpn import JTMPN
from baselines.jtvae.fastjt.jtnn_enc import JTNNEncoder
from baselines.jtvae.fastjt.mpn import MPN
from baselines.jtvae.fastjt.vocab import Vocab

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import rdkit.Chem as Chem
from rdkit.Chem import QED, Descriptors


class MolTreeFolder(object):
    def __init__(
        self,
        data,
        baseline_dir,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        assm=True,
        replicate=None,
        score=None,
        partition="train",
    ):
        assert data in ["moses", "zinc", "chembl"]
        self.data_folder = baseline_dir / "data" / "JTVAE" / data / partition
        self.vocab = baseline_dir / "data" / "JTVAE" / data / "jtvae_vocab.txt"
        self.vocab = Vocab([x.strip("\r\n ") for x in open(self.vocab)])
        self.name = data
        self.data_files = [fn for fn in os.listdir(self.data_folder)]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.score = score

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm, score=self.score)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=lambda x: x[0],
            )

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeDataset(Dataset):
    def __init__(self, data, vocab, assm=True, score=None):
        self.data = data
        self.vocab = vocab
        self.assm = assm
        self.score = score

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm, score=self.score)


def tensorize(tree_batch, vocab, assm=True, score=None):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)

    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    if score is None:
        molscore = None
    elif score == "penlogp":
        molscore = torch.Tensor([penlogp(smiles) for smiles in smiles_batch])
    elif score == "qed":
        molscore = torch.Tensor([qed(smiles) for smiles in smiles_batch])
    elif score == "drd2":
        molscore = torch.Tensor([drd2(smiles) for smiles in smiles_batch])
    else:
        raise NotImplementedError()

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx), molscore


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            try:
                node.wid = vocab.get_index(node.smiles)
            except:
                raise ValueError(node.smiles)
            tot += 1


def penlogp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)
    return score


def qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    score = QED.qed(mol)
    return score


def drd2(smiles):
    raise NotImplementedError()
