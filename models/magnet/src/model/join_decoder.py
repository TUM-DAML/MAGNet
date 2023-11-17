from copy import deepcopy
from typing import Union

import igraph
import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import scipy.sparse as sp
import torch
import torch_sparse
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from torch_geometric.nn import MLP
from torch_sparse import SparseTensor

from models.magnet.src.chemutils.rdkit_helpers import get_max_valency
from models.magnet.src.data.mol_module import collate_fn, get_atom_positional
from models.magnet.src.utils import extract_blockdiag_sparse, manual_batch_to_device


class JoinDecoder(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        dim_config = parent_ref.dim_config
        node_feat_dim = (
            dim_config["atom_id_dim"]
            + dim_config["atom_charge_dim"]
            + dim_config["atom_multiplicity_dim"]
            + dim_config["motif_feat_dim"]
            + dim_config["shape_gnn_dim"]
            + dim_config["shape_id_dim"]
            + dim_config["shape_multiplicity_dim"]
        )
        input_dim = node_feat_dim * 2 + dim_config["latent_dim"]
        self.mlp = MLP([input_dim, input_dim, input_dim // 2, 1])

    def forward(self, batch, z_graph, sample, parent_ref):
        if sample:
            join_outputs, batch = self.forward_inference(batch, z_graph, parent_ref)
        else:
            join_outputs = self.forward_train(batch, z_graph, parent_ref)
        return join_outputs, batch

    def forward_train(self, batch, z_graph, parent_ref):
        # From here, we will duplicate nodes and their features in the original graph to mimic connecting nodes
        num_nodes_in_shape = batch["num_nodes_in_shape"]
        nodes_in_shape = batch["nodes_in_shape"]
        atom_idx = batch["atom_idx"][nodes_in_shape]
        atom_charges = batch["atom_charges"][nodes_in_shape]
        # prepare features for each node
        atom_id_embedding = parent_ref.node_featurizer.atom_id_emb(atom_idx)
        atom_charge_embedding = parent_ref.node_featurizer.atom_charge_emb(atom_charges + 1)
        atom_positional_embedding = parent_ref.node_featurizer.atom_mult_emb(batch["mult_node_in_shape"].long() + 1)
        motif_per_shape_feats = torch.repeat_interleave(batch["feats_in_motif"], num_nodes_in_shape, dim=0)
        motif_feat_embeding = parent_ref.node_featurizer.motif_feat_transform(motif_per_shape_feats)
        shape_node_feats = torch.repeat_interleave(batch["shape_node_embeddings"], num_nodes_in_shape, dim=0)
        node_features = torch.cat(
            [
                atom_id_embedding,
                atom_charge_embedding,
                atom_positional_embedding,
                shape_node_feats,
                motif_feat_embeding,
            ],
            dim=-1,
        )

        # construct off-diagonal matrix with possible atom joins
        possible_joins = get_train_possible_joins(batch)
        edge_index = torch.nonzero(possible_joins).T

        # predict atom joins
        z_graph_repeated = torch.repeat_interleave(z_graph, batch["num_core_atoms_pre_join"], dim=0)
        input = torch.cat((node_features, z_graph_repeated), dim=1)
        input = input[edge_index.T].flatten(1)[:, : -z_graph.size(-1)]
        atom_join = self.mlp(input).squeeze(0)
        atom_join = SparseTensor.from_edge_index(edge_index, atom_join, sparse_sizes=possible_joins.shape)
        atom_join = _to_symmetric(atom_join)
        return dict(atom_join_logits=atom_join)

    def single_sample_inference(self, input, dataset):
        (
            shape_adj_scipy,
            join_logits,
            motif_mol_input,
            shape_classes,
            motif_feats,
            num_nodes_in_shape,
            nodes_in_shape,
            allowed_joins,
            n_counter,
        ) = input
        # so networkx is really mean... these are not sorted somehow :(
        motif_mol = nx.Graph()
        motif_mol.add_nodes_from(sorted(motif_mol_input.nodes(data=True)))
        motif_mol.add_edges_from(motif_mol_input.edges(data=True))

        motif_mol = nx.relabel_nodes(motif_mol, {k: k - n_counter for k in motif_mol.nodes})
        nodes_in_shape -= n_counter
        # prepare attributes for graph
        join_logits = join_logits.toarray()
        motif_feats = np.repeat(motif_feats.numpy(), num_nodes_in_shape.int().numpy(), axis=0)
        shape_id = torch.arange(len(num_nodes_in_shape)).repeat_interleave(num_nodes_in_shape).numpy()
        pos_in_shape = np.concatenate([np.arange(nn) for nn in num_nodes_in_shape])
        for k in motif_mol.nodes:
            motif_mol.nodes[k]["JoinAllowed"] = allowed_joins[k]
            motif_mol.nodes[k]["ShapeId"] = [shape_id[k].item()]
            motif_mol.nodes[k]["PosInShape"] = [pos_in_shape[k].item()]

        # In the following, we will contract join nodes through an nx.Graph
        join_map = dict()

        for k, (i, j) in enumerate(zip(shape_adj_scipy.row, shape_adj_scipy.col)):
            # no self joins allowed (may happen due to self-loops)
            if i == j:
                continue

            # find out possible joins according to correction criteria
            join_input = dict(
                motif_mol=motif_mol,
                num_nodes_in_shape=num_nodes_in_shape,
                allowed_joins=allowed_joins,
            )
            possible_joins, i_begin, j_begin = get_possible_joins(join_input, i, j, dataset)
            if possible_joins is None:
                continue
            edge_idxs = torch.nonzero(possible_joins).T
            pred_join = join_logits[i_begin + edge_idxs[0], j_begin + edge_idxs[1]]

            # hypernode adjacency prediction masks out joins that dont adhere to predicted join type
            # if none of them are available due to charge, valency, etc. constraints, skip
            if pred_join.max() == 0:
                continue
            best_join_idx = edge_idxs[:, pred_join.squeeze().argmax()].tolist()

            # maintain graph and joined list
            i_join = best_join_idx[0] + i_begin.item()
            j_join = best_join_idx[1] + j_begin.item()

            # check for previously joined nodes
            if i_join in join_map:
                i_join = join_map[i_join]
            if j_join in join_map:
                j_join = join_map[j_join]

            min_join = min(i_join, j_join)
            max_join = max(i_join, j_join)

            assert not motif_mol.nodes[min_join]["IsJoined"]
            assert not motif_mol.nodes[max_join]["IsJoined"]
            assert min_join != max_join

            # check whether the join we are performing is valid
            assert motif_mol.nodes[min_join]["AtomicSymbole"] == motif_mol.nodes[max_join]["AtomicSymbole"]
            assert motif_mol.nodes[min_join]["FormalCharge"] == motif_mol.nodes[max_join]["FormalCharge"]

            # maintain shape membership as well as position in shape
            motif_mol.nodes[min_join]["ShapeId"] = (
                motif_mol.nodes[min_join]["ShapeId"] + motif_mol.nodes[max_join]["ShapeId"]
            )
            motif_mol.nodes[min_join]["PosInShape"] = (
                motif_mol.nodes[min_join]["PosInShape"] + motif_mol.nodes[max_join]["PosInShape"]
            )

            # perform actual join
            if not motif_mol.nodes[min_join]["InRing"] and not motif_mol.nodes[max_join]["InRing"]:
                motif_mol.nodes[min_join]["IsJoined"] = True
            if len(motif_mol.nodes[min_join]["ShapeId"]) == 3:
                motif_mol.nodes[min_join]["IsJoined"] = True
            motif_mol = nx.contracted_nodes(motif_mol, min_join, max_join)
            join_map[max_join] = min_join

        atom_idx = torch.tensor([dataset.atom_to_id[cn[1]["AtomicSymbole"]] for cn in motif_mol.nodes.data()])
        atom_charges = torch.tensor([n[1]["FormalCharge"] for n in motif_mol.nodes.data()])

        motif_features, shape_class = [], []
        motif_filler = np.full((motif_feats.shape[1],), fill_value=-1)
        for node in motif_mol.nodes:
            mf, sc = [], []
            sc.append(shape_classes[node].item())
            mf.append(motif_feats[node])
            for k, v in join_map.items():
                if v == node:
                    mf.append(motif_feats[k])
                    sc.append(shape_classes[k])
            for _ in range(3 - len(mf)):
                mf.append(motif_filler)
                sc.append(-1)
            motif_features.append(np.concatenate(mf))
            shape_class.append(sc)

        atom_adj = SparseTensor.from_dense(torch.Tensor(nx.to_numpy_array(motif_mol, weight="BTDouble")))

        # nodes in shape tells us the shape -> atom mapping, update with joined atoms
        nodes_in_shape = np.array([n if n not in join_map.keys() else join_map[n] for n in nodes_in_shape.tolist()])
        # adjust indices for removed nodes
        removed_idx = np.array(list(join_map.keys()))
        nodes_in_shape = np.array([n - np.sum(removed_idx < n) for n in nodes_in_shape])
        mult_node_in_shape, mult_per_atom = get_atom_positional(
            atom_idx, nodes_in_shape, num_nodes_in_shape.int().tolist()
        )
        return dict(
            atom_idx=atom_idx,
            atom_charges=atom_charges,
            atom_adj=atom_adj.to_scipy(layout="coo"),
            shape_classes=torch.tensor(shape_class),
            motif_features=torch.tensor(np.array(motif_features)).float(),
            mult_node_in_shape=mult_node_in_shape,
            mult_per_atom=mult_per_atom,
            nodes_in_shape=nodes_in_shape,
        )

    def forward_inference(self, batch, z_graph, parent_ref):
        dataset = parent_ref.dataset_ref
        device = z_graph.device
        num_nodes = batch["num_nodes"]
        num_shape_nodes = batch["num_nodes_hgraph"].tolist()
        # prediction akin to taining
        join_pred = self.forward_train(batch, z_graph, parent_ref)
        join_logits = deepcopy(join_pred["atom_join_logits"].detach())
        # because it is a sparse tensor, we have to distinguish between "possible but unlikely" and "impossible"
        # shift probabilities by one s.t. possible (but unlikely) joins are strictly larger impossible joins
        new_vals = torch.sigmoid(join_logits.coo()[2]) + 1
        join_logits = join_logits.set_value(new_vals.squeeze(), layout="coo")

        shape_adj_scipy = batch["shape_adj"].to_scipy(layout="coo")
        shape_adj_scipy = sp.triu(shape_adj_scipy)

        # join into one big molecule
        is_ring = torch.repeat_interleave(torch.tensor(batch["is_ring_motif"]).to(device), batch["num_nodes_in_shape"])
        graph = igraph.Graph()
        for k in range(len(batch["atom_idx"])):
            graph.add_vertex(
                batch["atom_idx"][k].item(),
                AtomicSymbole=dataset.id_to_atom[batch["atom_idx"][k].item()],
                InRing=is_ring[k].item(),
                FormalCharge=batch["atom_charges"][k].item(),
                IsJoined=False,
            )
        row, col, vals = SparseTensor.from_scipy(sp.triu(batch["atom_adj"].to_scipy())).coo()
        for k in range(len(row)):
            graph.add_edge(row[k].item(), col[k].item(), BTDouble=vals[k].item())
        batch_mol = graph.to_networkx()

        all_inputs_single = zip(
            unravel_batch(shape_adj_scipy, num_shape_nodes),
            unravel_batch(join_logits, num_nodes),
            unravel_batch(batch_mol, num_nodes),
            unravel_batch(batch["shape_classes"].cpu(), num_nodes),
            unravel_batch(batch["motif_feats"].cpu(), num_shape_nodes),
            unravel_batch(batch["num_nodes_in_shape"].cpu(), num_shape_nodes),
            unravel_batch(batch["nodes_in_shape"], num_nodes),
            unravel_batch(batch["allowed_joins"].cpu(), num_nodes),
            np.cumsum(np.array([0] + batch["num_nodes"]))[:-1],
        )
        # this can infact be parallelized with more cores if we wanted to, for now lets keep it sequential
        outputs = Parallel(n_jobs=1)(delayed(self.single_sample_inference)(s, dataset) for s in all_inputs_single)

        # stack together outputs again
        batch_joined = collate_fn(outputs, inference=True)
        batch.update(batch_joined)
        nodes_in_shape = [o["nodes_in_shape"] for o in outputs]
        batch["nodes_in_shape"] = torch.tensor(
            np.concatenate([i + c for i, c in zip(nodes_in_shape, np.cumsum(np.array([0] + batch["num_nodes"][:-1])))])
        )
        manual_batch_to_device(batch, z_graph.device)
        return dict(), batch

    def calculate_loss(self, batch, decoder_outputs, dataset):
        _, _, preds = decoder_outputs["atom_join_logits"].coo()
        # construct target with off-diagonal entries
        target = (batch["nodes_in_shape"].view(-1, 1) == batch["nodes_in_shape"].view(1, -1)).int()
        target = target - torch.eye(target.size(0)).to(target.device)
        possible_joins = get_train_possible_joins(batch)
        target = target[possible_joins.bool()]

        pos_weight = (target == 0).sum() / (target == 1).sum()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_atom_joins = loss_fn(preds.squeeze(), target)
        join_acc = (torch.round(torch.sigmoid(preds.squeeze())) == target).float().mean()
        return dict(loss_joins=loss_atom_joins, join_acc=join_acc)


def _to_symmetric(sparse_tensor):
    sparse_tensor = torch_sparse.cat([sparse_tensor, sparse_tensor.t()], dim=-1)
    _, _, vals = sparse_tensor.coo()
    vals = vals.view(vals.size(0), 2, -1).mean(-2)
    sparse_tensor = sparse_tensor.set_value(vals, layout="coo")
    return sparse_tensor


def get_possible_joins(batch, r, c, dataset_ref):
    mol = batch["motif_mol"]
    dev = batch["num_nodes_in_shape"].device
    # overwrite join constraint if it is a cycle (bc we allow 3 joins for cycles)
    # maintain shape sizes to slice out relevant parts of the adjacency
    num_nodes_cum = torch.cat((torch.zeros((1,)).to(dev), torch.cumsum(batch["num_nodes_in_shape"], dim=0)))
    # get those intervals in which the to-be-joined shapes are
    r_start = num_nodes_cum[r].int()
    c_start = num_nodes_cum[c].int()
    # select atoms that we are working with
    r_nodes = [n for n in mol.nodes.data() if r in n[1]["ShapeId"]]
    c_nodes = [n for n in mol.nodes.data() if c in n[1]["ShapeId"]]
    # resort nodes here, as they might not appear in list in canonical order due to node contraction
    r_nodes = sorted(r_nodes, key=lambda x: x[1]["PosInShape"][x[1]["ShapeId"].index(r)])
    c_nodes = sorted(c_nodes, key=lambda x: x[1]["PosInShape"][x[1]["ShapeId"].index(c)])
    r_atom_type = torch.tensor([dataset_ref.atom_to_id[rn[1]["AtomicSymbole"]] for rn in r_nodes])
    c_atom_type = torch.tensor([dataset_ref.atom_to_id[cn[1]["AtomicSymbole"]] for cn in c_nodes])
    r_charge = torch.tensor([rn[1]["FormalCharge"] for rn in r_nodes])
    c_charge = torch.tensor([rn[1]["FormalCharge"] for rn in c_nodes])
    r_max_valency = get_max_valency(r_atom_type, r_charge, dataset_ref.id_to_atom).numpy()
    c_max_valency = get_max_valency(c_atom_type, c_charge, dataset_ref.id_to_atom).numpy()
    # define "average" valency, basically just a way to broadcast allowed valency
    allowed_valency = (r_max_valency[:, None] + c_max_valency[None, :]) / 2
    # does this join result in a node with too high valency?
    r_valency = np.array([sum([mol.get_edge_data(*e)["BTDouble"] for e in mol.edges(n[0])]) for n in r_nodes])
    c_valency = np.array([sum([mol.get_edge_data(*e)["BTDouble"] for e in mol.edges(n[0])]) for n in c_nodes])
    r_joined = np.array([rn[1]["IsJoined"] for rn in r_nodes]).astype(bool)
    c_joined = np.array([cn[1]["IsJoined"] for cn in c_nodes]).astype(bool)
    r_join_allowed = np.array([rn[1]["JoinAllowed"] for rn in r_nodes]).astype(bool)
    c_join_allowed = np.array([cn[1]["JoinAllowed"] for cn in c_nodes]).astype(bool)
    current_valency = r_valency[:, None] + c_valency[None, :]
    valency_ok = current_valency <= allowed_valency
    # nodes can only be joined once AND has to be an "end node" (if not cycle)
    r_joined = np.logical_or(r_joined, ~r_join_allowed)
    c_joined = np.logical_or(c_joined, ~c_join_allowed)
    # are the nodes of the same atom type? e.g. can not join O and C
    same_atoms = r_atom_type.view(-1, 1) == c_atom_type.view(1, -1)
    same_charges = r_charge.view(-1, 1) == c_charge.view(1, -1)
    same_atoms = np.logical_and(same_atoms, same_charges)
    # have the nodes been joined with other nodes already?
    same_atoms[r_joined] = False
    same_atoms[:, c_joined] = False
    possible_joins = np.logical_and(same_atoms, valency_ok)
    if possible_joins.sum() == 0:
        return None, r_start, c_start
    return possible_joins.int(), r_start, c_start


def get_train_possible_joins(batch):
    num_nodes_in_shape = batch["num_nodes_in_shape"]
    nodes_in_shape = batch["nodes_in_shape"]
    atom_idx = batch["atom_idx"][nodes_in_shape].int()

    # construct off-diagonal matrix with possible atom joins
    hypernode_adj = batch["hypernode_adj"]
    hypernode_targets = torch.repeat_interleave(hypernode_adj.to_dense(), num_nodes_in_shape, dim=0)
    hypernode_targets = torch.repeat_interleave(hypernode_targets, num_nodes_in_shape, dim=1)
    atom_idx_shifted = atom_idx + 1
    atom_idx_rep = (atom_idx_shifted.view(-1, 1) + atom_idx_shifted.view(1, -1)) / 2

    # entries in atom_idx_rep represent join identity, discard joins on unequal atoms
    same_atoms = atom_idx.view(-1, 1) == atom_idx.view(1, -1)
    atom_idx_rep = atom_idx_rep * same_atoms
    possible_joins = torch.logical_and(atom_idx_rep == hypernode_targets, hypernode_targets != 0)

    possible_joins[~batch["allowed_joins"].bool()] = False
    possible_joins[:, ~batch["allowed_joins"].bool()] = False
    return possible_joins


def unravel_batch(input: Union[SparseTensor, torch.Tensor, sp.coo_matrix], num_nodes: list) -> list:
    if isinstance(input, SparseTensor):
        output = extract_blockdiag_sparse(input.to_scipy(layout="csr"), num_nodes)
    elif isinstance(input, sp.coo_matrix):
        output = extract_blockdiag_sparse(input.tolil(), num_nodes)
        output = [o.tocoo() for o in output]
    elif isinstance(input, torch.Tensor):
        output = torch.split(input, num_nodes)
    elif isinstance(input, nx.Graph):
        counter = 0
        output = []
        for k, n in enumerate(num_nodes):
            output.append(input.subgraph(range(counter, counter + n)))
            counter += n
    else:
        print("invalid input type")
    return output
