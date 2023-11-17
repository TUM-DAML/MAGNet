import gc
import os
import pickle
import random

from rdkit import Chem
from torch.utils.data import Dataset

from baselines.hiervae.src.chemutils import get_leaves
from baselines.hiervae.src.mol_graph import MolGraph


class MoleculeDataset(Dataset):
    def __init__(self, data, vocab, avocab, batch_size):
        safe_data = []
        for mol_s in data:
            hmol = MolGraph(mol_s)
            ok = True
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles = attr["smiles"]
                ok &= attr["label"] in vocab.vmap
                for i, s in attr["inter_label"]:
                    ok &= (smiles, s) in vocab.vmap
            if ok:
                safe_data.append(mol_s)

        print(f"After pruning {len(data)} -> {len(safe_data)}")
        self.batches = [safe_data[i : i + batch_size] for i in range(0, len(safe_data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)


class MolPairDataset(Dataset):
    def __init__(self, data, vocab, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = zip(*self.batches[idx])
        x = MolGraph.tensorize(x, self.vocab, self.avocab)[:-1]  # no need of order for x
        y = MolGraph.tensorize(y, self.vocab, self.avocab)
        return x + y


class DataFolder(object):
    def __init__(self, name, batch_size, BASELINE_DIR, shuffle=True, partition="train", property=None):
        assert name in ["chembl", "moses", "zinc"]
        self.name = name
        self.data_folder = BASELINE_DIR / "data" / "HIERVAE" / name / "hiervae_preproc"
        if property is not None:
            self.data_folder = self.data_folder / property
        else:
            self.data_folder = self.data_folder / partition
        self.data_files = [fn for fn in os.listdir(self.data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partition = partition

    def __len__(self):
        # see preprocessing script
        return len(self.data_files) * 100

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                batches = pickle.load(f)

            if self.shuffle:
                random.shuffle(batches)  # shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()
