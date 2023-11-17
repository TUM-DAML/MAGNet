import argparse
import os
import pickle
from itertools import chain
from pathlib import PosixPath

import networkx as nx
import rdkit.Chem as Chem
import tqdm
from torch.multiprocessing import Pool

from src.chemutils.hypergraph import MolDecomposition
from src.utils import smiles_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--num_processes", type=int, default=15)
    parser.add_argument("--output_name", default="magnet_vocab.pkl")
    args = parser.parse_args()

    args.path = PosixPath(args.path)
    all_smiles = all_smiles_total = smiles_from_file(args.path)

    def process_func(smiles):
        mol_holder = MolDecomposition(smiles)
        mol_holder.create_motif_map()
        output_list = []
        for shape in mol_holder.id_to_fragment.values():
            frag_mol = Chem.MolFromSmiles(shape)
            adjacency = Chem.GetAdjacencyMatrix(frag_mol)
            graph = nx.Graph(adjacency)
            graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
            canonical_smiles = Chem.MolToSmiles(frag_mol, isomericSmiles=False, kekuleSmiles=True)
            output_list.append((graph_hash, canonical_smiles))
        return output_list

    # Multiprocessing but with Progress Bar
    with Pool(processes=args.num_processes) as p:
        outputs = []
        max_ = len(all_smiles)
        with tqdm.tqdm(total=max_) as pbar:
            for o in p.imap_unordered(process_func, all_smiles):
                pbar.update()
                outputs.extend(o)

    shape_dict = dict()
    for hash, shape in tqdm.tqdm(outputs):
        if hash in shape_dict.keys():
            if shape not in shape_dict[hash]:
                shape_dict[hash] = shape_dict[hash] + [shape]
        else:
            shape_dict[hash] = [shape]

    print("Total vocabulary size: ", len(list(chain([shapes for shapes in shape_dict.values()]))))

    fp = args.path.parent / args.output_name
    with open(fp, "wb") as f:
        pickle.dump(shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
