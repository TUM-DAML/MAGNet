import warnings
from copy import deepcopy
from typing import Union

import numpy as np
import rdkit.Chem as Chem
import torch
import torch.nn.functional as F
import torch_geometric
import torch_sparse
from torch.nn import GRU, CrossEntropyLoss, Embedding
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MLP, TransformerConv
from torch_sparse import SparseTensor

from models.magnet.src.chemutils.hypergraph import (
    is_all_cyclic,
    is_atom_cyclic_junction,
    is_cyclic_junction,
)
from models.magnet.src.chemutils.motif_helpers import atom_multiset_to_counts
from models.magnet.src.chemutils.rdkit_helpers import (
    MolFromGraph,
    compute_fingerprint,
    get_atom_charges,
    get_max_valency,
    smiles_to_array,
)
from models.magnet.src.data.utils import (
    get_atom_multiplicity_from_unsorted,
    get_atom_positional,
)
from models.magnet.src.model.transformer_utils import transformer_forward
from models.magnet.src.utils import calculate_balanced_acc, calculate_class_weights


class MotifDecoder(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        feat_sizes = parent_ref.feature_sizes
        dim_config = parent_ref.dim_config
        self.shape_graph_embedder = ShapeGraphEmbedder(
            feat_sizes, dim_config, parent_ref.layer_config["num_layers_hgraph"]
        )
        self.motif_atom_decoder = MotifAtomDecoder(feat_sizes, dim_config, self.shape_graph_embedder.output_dim)
        self.motif_bond_decoder = MotifBondDecoder(feat_sizes, dim_config, self.shape_graph_embedder.output_dim)

    def forward(self, batch, z_graph, sample, parent_ref):
        if sample:
            motif_outputs, batch = self.forward_inference(batch, z_graph, parent_ref)
        else:
            motif_outputs = self.forward_train(batch, z_graph, parent_ref)
        return motif_outputs, batch

    def forward_train(self, batch, z_graph, parent_ref):
        dataset = parent_ref.dataset_ref
        shape_embeddings = self.shape_graph_embedder(batch, z_graph, dataset, parent_ref.node_featurizer)
        motif_atom_logits = self.motif_atom_decoder(shape_embeddings, batch, dataset, parent_ref.node_featurizer)
        motif_bond_logits = self.motif_bond_decoder(
            shape_embeddings, batch, dataset, parent_ref.node_featurizer, self.motif_atom_decoder
        )
        return dict(motif_atom_logits=motif_atom_logits, motif_bond_logits=motif_bond_logits)

    def forward_inference(self, batch, z_graph, parent_ref):
        dataset = parent_ref.dataset_ref
        # construct adjacency blueprint -> we know edges from shape prediction
        reference_smiles = [dataset.shape_reference[s.item()] for s in batch["shape_node_idx"]]
        reference_mols = [Chem.MolFromSmiles(rs) for rs in reference_smiles]
        reference_adjs = [torch.tensor(Chem.rdmolops.GetAdjacencyMatrix(rm)) for rm in reference_mols]
        batch["motif_bond_target"] = torch.block_diag(*reference_adjs).to(z_graph.device)
        batch["motif_bond_target"] = SparseTensor.from_dense(batch["motif_bond_target"])
        shape_embeddings = self.shape_graph_embedder(batch, z_graph, dataset, parent_ref.node_featurizer)
        atom_idx_output, atom_charge_output = self.motif_atom_decoder.inference(
            shape_embeddings, batch, dataset, parent_ref.node_featurizer
        )
        bond_idx_output = self.motif_bond_decoder.inference(
            shape_embeddings,
            batch,
            dataset,
            parent_ref.node_featurizer,
            self.motif_atom_decoder,
            atom_idx_output,
            atom_charge_output,
        )
        # construct smiles:
        counter, motif_smiles = 0, []
        for nn in batch["num_nodes_in_shape"]:
            aidx = atom_idx_output[counter : counter + nn]
            cidx = atom_charge_output[counter : counter + nn]
            node_list = [dataset.id_to_atom[int(aid)] for aid in aidx]
            bidx = bond_idx_output[counter : counter + nn, counter : counter + nn]
            motif_mol = MolFromGraph(node_list, bidx, cidx - 1)
            assert motif_mol is not None
            motif_smiles.append(Chem.MolToSmiles(motif_mol))
            counter += nn
        batch = batch_from_motif_pred(batch, motif_smiles, z_graph.device, dataset)
        # Attention, here we will have more nodes than in the actual graph because of hypernode
        # All of what we are constructing here will later be joined
        return dict(), batch

    def calculate_loss(self, batch, decoder_outputs, dataset):
        (
            motif_atom_loss,
            motif_atom_acc,
            motif_charge_loss,
            motif_charge_acc,
        ) = self.motif_atom_decoder.calculate_loss(batch, decoder_outputs, dataset)
        motif_bond_loss, motif_bond_acc = self.motif_bond_decoder.calculate_loss(batch, decoder_outputs, dataset)
        loss_motifs = motif_atom_loss + motif_bond_loss + motif_charge_loss
        return dict(
            loss_motifs=loss_motifs,
            motif_atom_loss=motif_atom_loss,
            motif_bond_loss=motif_bond_loss,
            motif_atom_acc=torch.tensor(motif_atom_acc),
            motif_bond_acc=torch.tensor(motif_bond_acc),
            motif_charge_loss=motif_charge_loss,
            motif_charge_acc=torch.tensor(motif_charge_acc),
        )


def batch_from_motif_pred(batch, motif_smiles_pred, device, dataset):
    """
    Decode motif predictions into a batch that can be used for the next decoding step
    batch: previously decoded batch
    motif_pred: predicted motifs
    shape_node_idx: shape node indices
    device: device to put the tensors on
    dataset: reference to the dataset
    """
    atom_node_ids, atom_adj, atom_charges, num_nodes = [], [], [], []
    predicted_motifs, shape_classes, num_nodes_in_shape = [], [], []
    is_ring_motif, motif_feats, allowed_joins, atom_mult = [], [], [], []
    motif_smiles = []
    shape_node_idx = torch.split(batch["shape_node_idx"], batch["num_nodes_hgraph"].tolist())
    # chunk list of smiles into chunks of size batch["num_nodes_hgraph"]
    counter = 0
    motif_pred = []
    for nn in batch["num_nodes_hgraph"].tolist():
        motif_pred.append(motif_smiles_pred[counter : counter + nn])
        counter += nn
    for sample_pred, sample_node_idx in zip(motif_pred, shape_node_idx):
        num_nodes_sample = []
        for motif, node_idx in zip(sample_pred, sample_node_idx.tolist()):
            # decode all important information for later decoding steps
            motif_smiles.append(motif)
            motif_feats.append(torch.Tensor(compute_fingerprint(motif)))
            predicted_motifs.append(motif)

            _, atom_idx, adj, mol = smiles_to_array(motif, dataset.atom_to_id)
            ordering = list(Chem.rdmolfiles.CanonicalRankAtoms(mol))
            _, ordering = torch.sort(torch.tensor(ordering))
            atom_idx = atom_idx[ordering]
            adj = adj[ordering, :][:, ordering]
            atom_mult.append(get_atom_multiplicity_from_unsorted(atom_idx))
            is_ring = is_all_cyclic(mol)
            is_ring_motif.append(is_ring)
            if is_ring:
                allowed_joins.extend(np.ones_like(atom_idx).astype(bool).tolist())
            elif is_cyclic_junction(mol):
                joins_temp = np.array([0 if is_atom_cyclic_junction(a) else 1 for a in mol.GetAtoms()])
                joins_temp = joins_temp[ordering]
                allowed_joins.extend(joins_temp.tolist())
            else:
                node_degrees = np.array([a.GetDegree() for a in mol.GetAtoms()])[ordering]
                allowed_joins.extend((node_degrees == 1).tolist())
            atom_node_ids.append(atom_idx)
            atom_adj.append(torch.tensor(adj))
            shape_classes.append(torch.full(atom_idx.shape, fill_value=node_idx))
            atom_charges.append(get_atom_charges(mol)[ordering])
            num_nodes_in_shape.append(mol.GetNumAtoms())
            num_nodes_sample.append(mol.GetNumAtoms())
        num_nodes.append(sum(num_nodes_sample))

    batch["nodes_in_shape"] = torch.arange(sum(num_nodes_in_shape))
    batch["shape_classes"] = torch.cat(shape_classes).to(device)
    batch["atom_idx"] = torch.tensor(np.concatenate(atom_node_ids, axis=-1)).to(device)
    batch["atom_charges"] = torch.cat(atom_charges).to(device)
    batch["atom_adj"] = SparseTensor.from_dense(torch.block_diag(*atom_adj).to(device))
    batch["num_nodes_in_shape"] = torch.Tensor(num_nodes_in_shape).to(device).int()
    batch["num_nodes"] = num_nodes
    batch["num_core_atoms_pre_join"] = torch.tensor(num_nodes).to(device).int()
    batch["is_ring_motif"] = is_ring_motif
    batch["motif_feats"] = torch.stack(motif_feats).to(device)
    batch["feats_in_motif"] = torch.stack(motif_feats).to(device)
    batch["allowed_joins"] = torch.Tensor(allowed_joins).to(device)
    batch["motif_smiles"] = motif_smiles
    batch["atom_in_shape_mult"] = torch.cat([torch.tensor(a) for a in atom_mult]).to(device).long()
    batch["mult_node_in_shape"], batch["mult_per_atom"] = get_atom_positional(
        batch["atom_idx"], batch["nodes_in_shape"].numpy(), batch["num_nodes_in_shape"].int().tolist()
    )
    batch["mult_node_in_shape"] = torch.tensor(batch["mult_node_in_shape"]).to(device)
    batch["mult_per_atom"] = batch["mult_per_atom"].to(device)
    # we expect no joins / hypernodes here
    assert torch.all(batch["mult_per_atom"][:, 1] == -1)
    return batch


class MotifAtomDecoder(torch.nn.Module):
    def __init__(self, feat_sizes, dim_config, hidden_size):
        super().__init__()
        self.max_motif_len = feat_sizes["max_shape_size"]
        input_size = (
            hidden_size
            + dim_config["motif_seq_positional_dim"]
            + dim_config["atom_id_dim"]
            + dim_config["atom_charge_dim"]
        )
        self.to_atom = MLP([input_size, feat_sizes["num_atoms"] + 3])
        self.to_memory = MLP([hidden_size + feat_sizes["max_shape_size"], input_size, input_size])
        _transformer_layers = torch.nn.TransformerDecoderLayer(
            input_size, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.transformer = torch.nn.TransformerDecoder(_transformer_layers, num_layers=2)
        self.seq_pos_embedding = Embedding(feat_sizes["max_shape_size"], dim_config["motif_seq_positional_dim"])
        self.start_token = Embedding(1, dim_config["atom_id_dim"])
        self.start_token_charge = Embedding(1, dim_config["atom_charge_dim"])
        self.num_atoms = feat_sizes["num_atoms"]
        self.num_charges = 3
        self.split_dims = [self.num_atoms, self.num_charges]

    def forward(self, shape_embeddings, batch, dataset, node_featurizer):
        device = shape_embeddings.device
        # mask out padding for batched embedding input, insert back after
        is_padding = batch["motif_atom_target"] == dataset.sequence_tokens["pad_token"]
        motif_atom_target = deepcopy(batch["motif_atom_target"])
        # set sequence padding to C atoms s.t. we can use node featurizer in a batched way
        motif_atom_target[is_padding] = 0
        atom_id_emb = node_featurizer.atom_id_emb(motif_atom_target)
        atom_id_emb[is_padding] = -1
        motif_charge_target = deepcopy(batch["motif_charge_target"])
        motif_charge_target[is_padding] = 0
        atom_charge_emb = node_featurizer.atom_charge_emb(motif_charge_target + 1)
        atom_charge_emb[is_padding] = -1
        # get dummy start tokens because first step doesnt have previous input
        # once for the atom id
        start_token = self.start_token(torch.zeros(1).long().to(device))
        start_token = start_token.repeat_interleave(atom_id_emb.size(0), dim=0)
        atom_id_emb = torch.cat([start_token.unsqueeze(1), atom_id_emb], dim=1)
        # and once for the charge
        start_token = self.start_token_charge(torch.zeros(1).long().to(device))
        start_token = start_token.repeat_interleave(atom_charge_emb.size(0), dim=0)
        atom_charge_emb = torch.cat([start_token.unsqueeze(1), atom_charge_emb], dim=1)
        atom_positional = torch.arange(atom_id_emb.size(1)).unsqueeze(0)
        atom_positional = atom_positional.repeat(atom_id_emb.size(0), 1)
        atom_pos_emb = self.seq_pos_embedding(atom_positional.long().to(device))
        rnn_input = torch.cat([atom_id_emb, atom_pos_emb, atom_charge_emb], dim=-1)
        # repeat shape embeddings and concatenate to rnn input
        shape_emb_rep = shape_embeddings.unsqueeze(1).repeat_interleave(atom_id_emb.size(1), dim=1)
        rnn_input = torch.cat([rnn_input, shape_emb_rep], dim=-1)[:, :-1]
        _shape_emb = torch.repeat_interleave(shape_embeddings.contiguous(), batch["num_nodes_in_shape"], dim=0)
        transformer_output = transformer_forward(
            self.transformer,
            self.to_memory,
            batch["num_nodes_in_shape"],
            memory=_shape_emb,
            tgt_input=rnn_input,
            max_pos=self.max_motif_len,
        )
        motif_atom_logits = self.to_atom(transformer_output.reshape(-1, transformer_output.size(2)))
        return motif_atom_logits

    def inference(self, shape_embeddings, batch, dataset, node_featurizer):
        bs = shape_embeddings.size(0)
        device = shape_embeddings.device

        # construct input token
        batch["num_nodes_in_shape"] = [dataset.shape_to_size[s.item()] for s in batch["shape_node_idx"]]
        zero_dummy = torch.zeros(1).long().to(device)
        atom_start_token = self.start_token(zero_dummy)
        atom_start_token = atom_start_token.repeat_interleave(bs, dim=0)
        charge_start_token = self.start_token_charge(zero_dummy)
        charge_start_token = charge_start_token.repeat_interleave(bs, dim=0)
        atom_pos_emb = self.seq_pos_embedding(zero_dummy)
        atom_pos_emb = atom_pos_emb.repeat_interleave(bs, dim=0)
        input_embedding = torch.cat(
            [atom_start_token, atom_pos_emb, charge_start_token, shape_embeddings], dim=-1
        ).unsqueeze(1)
        # construct hidden state
        # rnn_hidden = shape_embeddings.unsqueeze(0)
        atom_prediction, charge_prediction, joint_dists = [], [], []
        for k in range(max(batch["num_nodes_in_shape"])):
            # prediction, rnn_hidden = self.rec_cell(input=input_embedding.unsqueeze(1), hx=rnn_hidden.contiguous())
            prediction = transformer_forward(
                self.transformer,
                self.to_memory,
                num_nodes=torch.full((input_embedding.size(0),), fill_value=k + 1),
                memory=torch.repeat_interleave(
                    shape_embeddings.contiguous(), torch.tensor(batch["num_nodes_in_shape"]).to(device), dim=0
                ),
                tgt_input=input_embedding,
                max_pos=self.max_motif_len,
                num_nodes_memory=batch["num_nodes_in_shape"],
                tgt_key_padding_mask=torch.zeros((input_embedding.size(0), k + 1)).to(device).bool(),
            )
            prediction = prediction[:, -1]
            atom_logits, charge_logits = torch.split(self.to_atom(prediction).squeeze(1), self.split_dims, dim=1)
            # mask out charges that we feel are not allowed
            input_token, charge_token, joint_dist = correct_atoms(charge_logits, atom_logits, batch, dataset, k)
            atom_prediction.append(input_token)
            charge_prediction.append(charge_token)
            joint_dists.append(joint_dist)
            atom_id_emb = node_featurizer.atom_id_emb(input_token)
            atom_charge_emb = node_featurizer.atom_charge_emb(charge_token)
            atom_pos_emb = self.seq_pos_embedding(torch.full((1,), fill_value=(k + 1)).long().to(device))
            atom_pos_emb = atom_pos_emb.repeat_interleave(bs, dim=0)
            input_embedding_new = torch.cat([atom_id_emb, atom_pos_emb, atom_charge_emb, shape_embeddings], dim=-1)
            input_embedding = torch.cat((input_embedding, input_embedding_new.unsqueeze(1)), dim=1)

        atom_prediction = torch.stack(atom_prediction).T
        # dimensions: num_shapes x max_seq_len x num_atoms x num_charges
        joint_dists = torch.stack(joint_dists).transpose(1, 0)
        charge_prediction = torch.stack(charge_prediction).T
        atom_idx_output, atom_charge_output = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for j in range(atom_prediction.size(0)):
            # CORRECT FOR REQUIRED JOINS HERE, WE CAN ALSO BAIL OUT THOUGH
            num_nodes = batch["num_nodes_in_shape"][j]
            pred_atom = atom_prediction[j][:num_nodes]
            pred_charge = charge_prediction[j][:num_nodes]
            pred_joint = joint_dists[j][:num_nodes]
            pred_charge, pred_atom = ensure_joins(batch, dataset, pred_atom, pred_charge, pred_joint, j)
            atom_idx_output = torch.cat([atom_idx_output, pred_atom])
            atom_charge_output = torch.cat([atom_charge_output, pred_charge])
        return atom_idx_output, atom_charge_output

    def calculate_loss(self, batch, decoder_outputs, dataset):
        atom_output, charge_output = torch.split(decoder_outputs["motif_atom_logits"], self.split_dims, dim=1)
        atom_target = batch["motif_atom_target"].reshape(-1)
        charge_target = batch["motif_charge_target"].reshape(-1)
        pad_token = dataset.sequence_tokens["pad_token"]
        atom_mask = atom_target != pad_token
        charge_target[atom_mask] = charge_target[atom_mask] + 1
        # calculate loss and accuracy for atom types
        weight = calculate_class_weights(atom_target[atom_mask], atom_output.size(1))
        loss_motif_atoms = CrossEntropyLoss(ignore_index=pad_token, weight=weight)(atom_output, atom_target)
        acc_motif_atoms = calculate_balanced_acc(atom_target[atom_mask], atom_output[atom_mask])
        # calculate loss and accuracy for atom charges
        weight = calculate_class_weights(charge_target[atom_mask], charge_output.size(1))
        loss_motif_charges = CrossEntropyLoss(ignore_index=pad_token, weight=weight)(charge_output, charge_target)
        acc_motif_charges = calculate_balanced_acc(charge_target[atom_mask], charge_output[atom_mask])
        return loss_motif_atoms, acc_motif_atoms, loss_motif_charges, acc_motif_charges


class MotifBondDecoder(torch.nn.Module):
    def __init__(self, feat_sizes, dim_config, hidden_size):
        super().__init__()
        input_size = (
            hidden_size
            + (dim_config["motif_seq_positional_dim"] + dim_config["atom_id_dim"] + dim_config["atom_charge_dim"]) * 2
        )
        self.bond_mlp = MLP([input_size, input_size // 2, feat_sizes["atom_adj_feat_size"]])

    def forward(self, shape_embeddings, batch, dataset, node_featurizer, motif_atom_decoder):
        row, col, _ = batch["motif_bond_target"].coo()
        edges_per_shape = torch.clamp(batch["motif_bond_target"].to_dense(), 0, 1).sum(-1)
        edges_per_shape = torch.split(edges_per_shape, batch["num_nodes_in_shape"].tolist())
        edges_per_shape = torch.tensor([bt.sum() for bt in edges_per_shape]).to(shape_embeddings.device)
        atom_idx_emb = node_featurizer.atom_id_emb(batch["motif_atoms"])
        atom_charge_emb = node_featurizer.atom_charge_emb(batch["motif_charges"] + 1)
        atom_pos = torch.cat([torch.arange(1, nn + 1) for nn in batch["num_nodes_in_shape"]])
        atom_pos_embedding = motif_atom_decoder.seq_pos_embedding(atom_pos.to(shape_embeddings.device))
        atom_feats = torch.cat([atom_idx_emb, atom_charge_emb, atom_pos_embedding], dim=1)
        shape_emb_rep = shape_embeddings.repeat_interleave(edges_per_shape.int(), dim=0)
        input = torch.cat([atom_feats[row], atom_feats[col], shape_emb_rep], dim=1)
        bond_logits = self.bond_mlp(input)
        # make into sparse tensor and make symmetric
        edge_index = torch.stack([row, col], dim=0)
        bond_logits = SparseTensor.from_edge_index(edge_index, bond_logits)
        bond_logits = _to_symmetric(bond_logits)
        return bond_logits

    def inference(self, shape_embeddings, batch, dataset, node_featurizer, motif_atom_decoder, atom_idx, atom_charges):
        batch["motif_atoms"] = atom_idx.long()
        batch["motif_charges"] = atom_charges.long() - 1
        batch["num_nodes_in_shape"] = torch.tensor(batch["num_nodes_in_shape"])
        bond_logits = self(shape_embeddings, batch, dataset, node_featurizer, motif_atom_decoder)
        bond_logits = bond_logits.to_dense()
        triu_mask = torch.triu(torch.ones(bond_logits.shape[:2]))
        bond_logits = bond_logits * triu_mask.unsqueeze(-1).to(atom_idx.device)
        bond_logits = SparseTensor.from_dense(bond_logits)

        row, col, bond_vals = bond_logits.coo()
        sorted_idx = torch.sort(bond_vals.max(-1)[0])[1].long()
        bond_vals = bond_vals[sorted_idx]
        row, col = row[sorted_idx], col[sorted_idx]
        bond_idx_output = batch["motif_bond_target"].to_dense()
        # correct for valency, can we even add this large bond type?
        for i, j, v in zip(row, col, bond_vals):
            # -1 because we do not take into account the current edge
            current_valence_i = (bond_idx_output.sum(-1) - 1)[i]
            max_valence_i = get_max_valency(atom_idx[i].view(-1), atom_charges[i].view(-1) - 1, dataset.id_to_atom)
            current_valence_j = (bond_idx_output.sum(-1) - 1)[j]
            max_valence_j = get_max_valency(atom_idx[j].view(-1), atom_charges[j].view(-1) - 1, dataset.id_to_atom)
            capacity_i = (max_valence_i - current_valence_i).squeeze()
            capacity_j = (max_valence_j - current_valence_j).squeeze()
            capacity = torch.min(capacity_i, capacity_j)
            assert capacity > 0
            bond_type = v[: capacity.int()].argmax() + 1
            bond_idx_output[i, j] = bond_type
            bond_idx_output[j, i] = bond_type
        return bond_idx_output

    def calculate_loss(self, batch, decoder_outputs, dataset):
        _, _, bond_output = decoder_outputs["motif_bond_logits"].coo()
        _, _, bond_target = batch["motif_bond_target"].coo()
        bond_target = bond_target - 1
        weight = calculate_class_weights(bond_target, bond_output.size(1))
        motif_bond_loss = CrossEntropyLoss(weight=weight)(bond_output, bond_target.long())
        acc_motif_bonds = calculate_balanced_acc(bond_target, bond_output)
        return motif_bond_loss, acc_motif_bonds


def ensure_joins(batch, dataset, pred_atom, pred_charge, pred_joint, j):
    device = pred_atom.device
    shape_idx = batch["shape_node_idx"][j]
    allowed_join_idx = dataset.shape_to_join_idx[shape_idx.item()]
    current_atom_counts = atom_multiset_to_counts(
        pred_atom[allowed_join_idx], [len(allowed_join_idx)], len(dataset.id_to_atom)
    ).squeeze()
    required_atom_counts = batch["hypernode_counts_in_shape"][j].squeeze()
    # since rings are allowed to have double joins, we can only ensure at least 1
    if dataset.shape_type[shape_idx.item()] == "ring":
        required_atom_counts = torch.clamp(required_atom_counts, max=1)
    # mask out those atoms that can not be joins
    join_mask = torch.zeros_like(pred_joint).bool().to(device)
    join_mask[allowed_join_idx] = True
    pred_joint[~join_mask] = 0
    while torch.any(current_atom_counts < required_atom_counts):
        pred_joint_copy = pred_joint.clone()
        # mask out those atoms that can not be changed because they are already
        # required for the hypernode join multiset
        fixed_atom_types = current_atom_counts <= required_atom_counts
        # permit "taking away" from aprts of the multiset that are of minimal size
        pred_joint_copy[fixed_atom_types[pred_atom]] = 0

        missing_atoms = ((required_atom_counts - current_atom_counts) > 0).nonzero()
        missing_atoms = missing_atoms.squeeze(-1).tolist()

        # taking last entry will prioritize rare missing atom joins
        ma = missing_atoms[-1]
        argmax_idx = find_argmax_in_tensor(pred_joint_copy[:, ma]).squeeze()
        # TODO future: we currently calculate the "likelihood", should we use the "posterior", i.e. reweight probabilities?
        if torch.isclose(pred_joint_copy[argmax_idx[0], ma, argmax_idx[1]], torch.zeros((1,)).to(device)):
            break
        pred_atom[argmax_idx[0]] = ma
        pred_charge[argmax_idx[0]] = argmax_idx[1]
        current_atom_counts = atom_multiset_to_counts(
            pred_atom[allowed_join_idx], [len(allowed_join_idx)], len(dataset.id_to_atom)
        ).squeeze()
    return pred_charge, pred_atom


def find_argmax_in_tensor(input):
    max_idx = torch.argmax(input.reshape(-1))
    zero_tens = torch.zeros_like(input.reshape(-1))
    zero_tens[max_idx] = 1
    zero_tens = zero_tens.reshape(input.size())
    return zero_tens.nonzero()


def correct_atoms(charge_logits, atom_logits, batch, dataset, k):
    device = atom_logits.device

    # prepare a list of nodes that are allowed to have negative and positive charges
    neg_charges_ok = torch.tensor([atom in ["O", "S"] for atom in dataset.atom_to_id.keys()]).to(device)
    pos_charges_ok = torch.tensor([atom in ["N", "S"] for atom in dataset.atom_to_id.keys()]).to(device)
    charge_prob = torch.nn.Softmax(dim=1)(charge_logits)
    atom_prob = torch.nn.Softmax(dim=1)(atom_logits)
    joint_dist = charge_prob.unsqueeze(1) * atom_prob.unsqueeze(2)
    # allow only intended ions
    joint_dist[:, ~neg_charges_ok, 0] = 0
    joint_dist[:, ~pos_charges_ok, -1] = 0

    # construct valency for all possible atom-charge combinations
    charge_mesh = torch.arange(-1, 2).repeat(len(dataset.atom_to_id))
    atom_mesh = torch.arange(len(dataset.atom_to_id)).repeat_interleave(3)
    max_valency = get_max_valency(atom_mesh, charge_mesh, dataset.id_to_atom).view(-1, 3)
    max_valency = max_valency.to(device).unsqueeze(0).repeat(joint_dist.size(0), 1, 1)
    # construct current valency (only single bonds)
    valencies_per_shape = torch.split(batch["motif_bond_target"].sum(-1), batch["num_nodes_in_shape"])
    valencies_per_shape = pad_sequence(valencies_per_shape, batch_first=True, padding_value=8)[:, k]
    valencies_per_shape = valencies_per_shape.view(-1, 1, 1).repeat(1, joint_dist.size(1), joint_dist.size(2))

    # single out atoms that can not happen in the current valency setting
    violated_valency = valencies_per_shape > max_valency
    joint_dist[violated_valency] = 0

    # pick "most likely" atom + charge combination after constraints
    atom_idx, charge_idx = [], []
    for jd in joint_dist:
        argmax_idx = find_argmax_in_tensor(jd)
        atom_idx.append(argmax_idx[:, 0].item())
        charge_idx.append(argmax_idx[:, 1].item())
    atom_idx = torch.tensor(atom_idx).to(device)
    charge_idx = torch.tensor(charge_idx).to(device)

    # we will reuse joint dist for the coming steps but for join atoms,
    # so they can not be at full valency but have to let at least one bond open
    violated_valency = valencies_per_shape > (max_valency - 1)
    joint_dist[violated_valency] = 0
    return atom_idx, charge_idx, joint_dist


def _to_symmetric(sparse_tensor):
    sparse_tensor = torch_sparse.cat([sparse_tensor, sparse_tensor.t()], dim=-1)
    _, _, vals = sparse_tensor.coo()
    vals = vals.view(vals.size(0), 2, -1).mean(-2)
    sparse_tensor = sparse_tensor.set_value(vals, layout="coo")
    return sparse_tensor


class ShapeGraphEmbedder(torch.nn.Module):
    def __init__(self, feat_sizes, dim_config, num_layers):
        super().__init__()
        input_dim = dim_config["shape_id_dim"] + dim_config["shape_multiplicity_dim"]
        gnn_dim = dim_config["shape_gnn_dim"]
        self.gnn_layers = []
        for i in range(num_layers):
            self.gnn_layers.append(
                (
                    TransformerConv(
                        gnn_dim if i > 0 else input_dim,
                        gnn_dim,
                        edge_dim=dim_config["atom_id_dim"],
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
            self.gnn_layers.append(torch.nn.ReLU())
        self.output_dim = input_dim + gnn_dim + dim_config["latent_dim"]
        self.gnn_layers = torch_geometric.nn.Sequential("x, edge_index, edge_attr", self.gnn_layers)

    def forward(self, batch, z_graph, dataset, node_featurizer):
        shape_id_embedding = node_featurizer.shape_id_emb(batch["shape_node_idx"])
        shape_mult_embedding = node_featurizer.shape_mult_emb(batch["shape_node_mult"])
        x = torch.cat([shape_id_embedding, shape_mult_embedding], dim=-1)
        # at inference time, shape adjacency will be dense
        row, col, _ = batch["shape_adj"].coo()
        edge_index = torch.stack([row, col], dim=0)
        join_identities = batch["hypernode_adj"].coo()[2]
        join_id_emb = node_featurizer.atom_id_emb(join_identities - 1)
        node_embeddings = self.gnn_layers(x, edge_index, join_id_emb)
        x = torch.cat((x, node_embeddings), dim=1)
        batch["shape_node_embeddings"] = x
        z_graph_rep = torch.repeat_interleave(z_graph, batch["num_nodes_hgraph"], dim=0)
        x = torch.cat((x, z_graph_rep), dim=-1)
        return x
