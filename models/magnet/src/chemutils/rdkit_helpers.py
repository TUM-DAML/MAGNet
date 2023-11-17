from copy import deepcopy
from typing import Union

import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.DataStructs as DataStructs
import torch
from rdkit import RDLogger
from torch_sparse import SparseTensor

from baselines.magnet.src.chemutils.constants import BOND_LIST

RDLogger.DisableLog("rdApp.*")


def get_max_valency(atom_idx, atom_charge, class_to_atom):
    node_symbols = [class_to_atom[a.item()] for a in atom_idx]
    periodic_table = Chem.GetPeriodicTable()
    max_valence = []
    for n in node_symbols:
        if n == "S":
            valence = 6
        elif n == "P":
            valence = 5
        else:
            valence = periodic_table.GetDefaultValence(n)
        max_valence.append(valence)
    max_valence = torch.Tensor(max_valence).to(atom_idx.device)
    max_valence += atom_charge
    return max_valence


def get_atom_charges(mol):
    charge = torch.Tensor([a.GetFormalCharge() for a in mol.GetAtoms()])
    return charge.int()


def smiles_to_array(sm, atom_to_class, return_charges=False):
    graph, mol = build_mol_graph(sm)
    adj = nx.to_numpy_array(graph)

    # retrieve node features and convert to one-hot encoding
    node_feats = [v[0] for (_, v) in nx.get_node_attributes(graph, "label").items()]
    node_feats = np.array([atom_to_class[nf] for nf in node_feats]).astype(np.long)

    # extend adjacency to edge tensor
    edge_feats = np.zeros((adj.shape[0], adj.shape[1]))
    # converting the graph to undirected saves unneccesary for-loop steps
    for (k1, k2), v in nx.get_edge_attributes(graph.to_undirected(), "label").items():
        edge_feats[k1, k2] = v + 1
        edge_feats[k2, k1] = v + 1

    # check for symmetric adjacencies
    assert np.array_equal(adj, adj.T)
    # check if we have features for all nodes
    assert node_feats.shape[0] == adj.shape[0] == adj.shape[1]
    # sanity check for edge feature generation
    assert np.array_equal(np.clip(edge_feats, 0, 1), adj)
    if return_charges:
        return adj, node_feats, edge_feats, mol, get_atom_charges(mol)
    return adj, node_feats, edge_feats, mol


def build_mol_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
    graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
    for atom in mol.GetAtoms():
        graph.nodes[atom.GetIdx()]["label"] = (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
        )

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        btype = BOND_LIST.index(bond.GetBondType())
        graph[a1][a2]["label"] = btype
        graph[a2][a1]["label"] = btype
    return graph, mol


def get_motif_features(motif_indicator, dataset):
    motif_features = []
    for m in motif_indicator:
        if m == -1:
            motif_features.append(torch.full((dataset.motif_feat_size,), -1))
        else:
            motif_smiles = dataset.class_to_motif[m + len(dataset.atoms)]
            motif_features.append(torch.Tensor(dataset.motif_to_rdkitfeats[motif_smiles]))
    return motif_features


def compute_fingerprint(input: Union[str, Chem.rdchem.Mol]) -> np.array:
    if isinstance(input, str):
        mol = Chem.MolFromSmiles(input)
    else:
        mol = deepcopy(input)
    top_feats = np.packbits(Chem.RDKFingerprint(mol, fpSize=2048)) / 255
    circ_feats = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(circ_feats, array)
    circ_feats = array
    mol_fingerprint = np.concatenate([top_feats, circ_feats])
    return mol_fingerprint


def pred_to_mols(sampled_mols, class_to_atom_map):
    nodes = sampled_mols["atom_idx"]
    num_nodes = sampled_mols["num_nodes"]
    atom_charges = sampled_mols["atom_charges"]
    bonds = sampled_mols["atom_adj"]
    # extract single molecules from batch
    bonds = extract_single_graphs(num_nodes, bonds.detach().cpu(), flatten=False)
    nodes = torch.split(nodes.cpu(), num_nodes)
    atom_charges = torch.split(atom_charges.cpu(), num_nodes)
    mols = []
    for node_set, e, ac in zip(nodes, bonds, atom_charges):
        node_set = [class_to_atom_map[n.item()] for n in node_set]
        mols.append(MolFromGraph(node_set, e, ac))
    return mols


def MolFromGraph(node_list, adjacency_matrix, atom_charges):
    """
    from: https://stackoverflow.com/questions/51195392/smiles-from-graph
    """
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        a.SetFormalCharge(atom_charges[i].int().item())
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix.int()):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            else:
                bond_type = BOND_LIST[bond - 1]
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    # catch invalid atom here
    except ValueError as e:
        print(e)
        mol = None
    except RuntimeError:
        print("Something went wrong with the edges")
        mol = None
    return mol


def extract_single_graphs(num_nodes, edges, flatten=True):
    """
    Extract disconnected components from batched adjacency
    """
    out = []
    n_start = 0
    for n_nodes in num_nodes:
        upper = n_start + n_nodes
        component = edges[n_start:upper, n_start:upper]
        if isinstance(component, SparseTensor):
            component = component.to_dense().squeeze()
        if flatten:
            component = component.flatten()
        out.append(component)
        n_start += n_nodes
    if flatten:
        out = torch.cat(out)
    return out
