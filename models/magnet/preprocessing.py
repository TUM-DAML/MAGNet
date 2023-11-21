import argparse
import bz2
import os
import pickle
from itertools import chain
from pathlib import PosixPath

import _pickle as cPickle
import networkx as nx
import rdkit.Chem as Chem
import tqdm
from interruptingcow import timeout
from torch.multiprocessing import Pool

import argparse
import bz2
import os
import pickle
from itertools import chain
from pathlib import PosixPath

import _pickle as cPickle
import networkx as nx
import rdkit.Chem as Chem
import tqdm
from interruptingcow import timeout
from torch.multiprocessing import Pool

from src.chemutils.hypergraph import MolDecomposition
from models.global_utils import DATA_DIR, SMILES_DIR
from models.magnet.src.utils import smiles_from_file


def process_func(input):
    output_dir, i, smiles = input

    # decompose molecule and save for training
    with timeout(20, exception=RuntimeError):
        mol_holder = MolDecomposition(smiles)
    fp = output_dir / f"{i:06d}.pbz2"
    with bz2.BZ2File(fp, "w") as f:
        cPickle.dump((i, smiles, mol_holder), f)

    # extract distinct shapes
    mol_holder.create_motif_map()
    output_list = []
    for shape in mol_holder.id_to_fragment.values():
        frag_mol = Chem.MolFromSmiles(shape)
        adjacency = Chem.GetAdjacencyMatrix(frag_mol)
        graph = nx.Graph(adjacency)
        graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
        canonical_smiles = Chem.MolToSmiles(frag_mol, isomericSmiles=False, kekuleSmiles=True)
        output_list.append((graph_hash, canonical_smiles))
    return output_list, mol_holder.mol.GetNumAtoms()


def run_magnet_preproc(dataset_name, num_processes):
    smiles_path = SMILES_DIR / dataset_name
    data_path = DATA_DIR / "MAGNET" / dataset_name
    partitions = ["train.txt", "val.txt", "test.txt"]
    vocabulary_name = "magnet_vocab.pkl"

    decomp_outputs = []

    for part in partitions:
        output_dir = data_path / part.split(".")[0]
        output_dir.mkdir(parents=True, exist_ok=True)

        all_smiles = smiles_from_file(smiles_path / part)
        all_smiles = [(output_dir, i, sm) for (i, sm) in enumerate(all_smiles)]

        with Pool(processes=num_processes) as p:
            max_ = len(all_smiles)
            with tqdm.tqdm(total=max_, desc="Decomposing " + part) as pbar:
                for o in p.imap_unordered(process_func, all_smiles):
                    pbar.update()
                    decomp_outputs.append(o)

    molecule_sizes = [d[1] for d in decomp_outputs]
    max_mol_size = max(molecule_sizes)
    min_mol_size = min(molecule_sizes)

    outputs = [d[0] for d in decomp_outputs]
    shape_dict = dict()
    max_num_shapes = len(max(outputs, key=len))
    outputs = list(chain(*outputs))
    for hash, shape in tqdm.tqdm(outputs, desc="Vocabulary Construction"):
        if hash in shape_dict.keys():
            if shape not in shape_dict[hash]:
                shape_dict[hash] = shape_dict[hash] + [shape]
        else:
            shape_dict[hash] = [shape]

    print("Almost done, just getting some vocabulary stats now...")
    shape_sizes = []
    for key, value in shape_dict.items():
        num_atoms = Chem.MolFromSmiles(value[0]).GetNumAtoms()
        shape_sizes.append(num_atoms)
    max_shape_size = max(shape_sizes)

    shape_dict["stats"] = dict(
        max_mol_size=max_mol_size,
        min_mol_size=min_mol_size,
        max_shape_size=max_shape_size,
        max_num_shapes=max_num_shapes,
    )

    print("Total vocabulary size: ", len(list(chain([shapes for shapes in shape_dict.values()]))))

    fp = data_path / vocabulary_name
    with open(fp, "wb") as f:
        pickle.dump(shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("MAGNet preprocessing done.")
