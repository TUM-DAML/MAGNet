from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from torch.nn import GRU, CrossEntropyLoss
from torch_geometric.nn import MLP

from models.magnet.src.chemutils.rdkit_helpers import get_max_valency
from models.magnet.src.model.transformer_utils import transformer_forward
from models.magnet.src.utils import calculate_balanced_acc, calculate_class_weights


class LeafDecoder(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        dim_config = parent_ref.dim_config
        feature_sizes = parent_ref.feature_sizes
        node_feat_size = parent_ref.node_featurizer.output_dim
        input_size = node_feat_size * 2 + dim_config["atom_multiplicity_dim"] + dim_config["atom_id_dim"]
        hidden_size = (
            dim_config["latent_dim"]
            + dim_config["motif_feat_dim"]
            + dim_config["shape_gnn_dim"]
            + dim_config["shape_id_dim"]
            + dim_config["shape_multiplicity_dim"]
        )
        output_dim = feature_sizes["atom_adj_feat_size"] + 1 + feature_sizes["num_atoms"]
        feature_dim = dim_config["leaf_rnn_hidden"]
        transformer_dim = feature_dim
        self.input_to_hidden = MLP([hidden_size, hidden_size // 2, feature_dim], plain_last=False)
        self.to_target = MLP([input_size + feature_dim, feature_dim, transformer_dim])
        self.to_memory = MLP([feature_dim + feature_sizes["max_shape_size"], transformer_dim])
        self.leaf_classifier = MLP([transformer_dim, feature_dim // 2, output_dim])
        self.num_edges = feature_sizes["atom_adj_feat_size"]
        self.num_atoms = feature_sizes["num_atoms"] + 1
        # categorial over number of atoms, none_token (0), start_token (-1), padding_token (-2)
        self.output_emb = torch.nn.Embedding(feature_sizes["num_atoms"] + 1 + 1 + 1, dim_config["atom_id_dim"])

        _transformer_layers = torch.nn.TransformerDecoderLayer(
            transformer_dim, nhead=2, dim_feedforward=256, batch_first=True
        )
        self.transformer = torch.nn.TransformerDecoder(_transformer_layers, num_layers=2)
        self.max_motif_len = feature_sizes["max_shape_size"]

    def forward(self, batch, z_graph, sample, parent_ref):
        if sample:
            leaf_outputs, batch = self.forward_inference(batch, z_graph, parent_ref)
        else:
            leaf_outputs = self.forward_train(batch, z_graph, parent_ref)
        return leaf_outputs, batch

    def forward_train(self, batch, z_graph, parent_ref):
        dataset = parent_ref.dataset_ref
        # extract embeddings from batch as deepcopy does not support gradient information copying
        shape_node_embeddings = batch.pop("shape_node_embeddings")
        batch_core = deepcopy(batch)
        batch_core, node_feats = self.extract_core_mol(batch_core, parent_ref.node_featurizer)
        batch_core["shape_node_embeddings"] = shape_node_embeddings
        rnn_input, hidden_feats = self.prepare_features(
            batch_core, z_graph, parent_ref.graph_encoder, parent_ref.node_featurizer, node_feats
        )
        rnn_input = self.input_padding(rnn_input, batch_core, hidden_feats)
        # prepare ground truth outputs to feed back to RNN at next step
        padding_token = dataset.sequence_tokens["pad_token"]
        outputs = batch_core["leaf_target"][0]
        start_tokens = torch.full((outputs.size(0), 1), fill_value=-1, device=z_graph.device)
        outputs_shifted = torch.cat((start_tokens, outputs), dim=1)[:, :-1]
        # replace padding token
        outputs_shifted[outputs_shifted == padding_token] = -2
        outputs_flattened = outputs_shifted.flatten()
        outputs_embedding_flattened = self.output_emb(outputs_flattened.long() + 2)
        outputs_embedding = outputs_embedding_flattened.view(outputs.size(0), outputs.size(1), -1)
        # concatenate input with teacher-forced output of last step
        rnn_input = torch.cat((rnn_input, outputs_embedding), dim=-1)
        # leaf_logits, _ = self.rec_cell(input=rnn_input, hx=hidden_feats.unsqueeze(0).contiguous())
        _shape_emb = torch.repeat_interleave(hidden_feats.contiguous(), batch["num_nodes_in_shape"], dim=0)
        bs, seq_len, dim = rnn_input.size()
        rnn_input = self.to_target(rnn_input.view(-1, dim)).reshape([bs, seq_len, -1])
        leaf_logits = transformer_forward(
            self.transformer,
            self.to_memory,
            batch["num_nodes_in_shape"],
            memory=_shape_emb,
            tgt_input=rnn_input,
            max_pos=self.max_motif_len,
        )

        leaf_logits = self.leaf_classifier(leaf_logits.reshape(-1, leaf_logits.size(-1)))
        return dict(leaf_logits=leaf_logits)

    def input_padding(self, input, batch, hidden_feats):
        dev = batch["atom_idx"].device
        maxs = max(batch["num_nodes_in_shape"])
        input = [torch.cat((ri, torch.zeros((maxs - ri.size(0), ri.size(1)), device=dev)), dim=0) for ri in input]
        input = torch.stack(input)
        input = torch.cat((input, hidden_feats.unsqueeze(1).repeat(1, input.size(1), 1)), dim=-1)
        return input

    def extract_core_mol(self, batch, node_featurizer):
        core_mol_mask = batch["shape_classes"][:, 0] != -1
        node_feats = node_featurizer(batch)["node_feats"][core_mol_mask]
        batch["atom_adj"] = batch["bond_adj"][core_mol_mask, core_mol_mask]
        new_node_idx = torch.arange(batch["atom_idx"].size(0)).cuda()[core_mol_mask].tolist()
        batch["nodes_in_shape"] = torch.tensor([new_node_idx.index(nis.item()) for nis in batch["nodes_in_shape"]])
        return batch, node_feats

    def prepare_features(self, batch, z_graph, graph_encoder, node_featurizer, node_feats):
        # TODO future: should ions be an atom class in the future? This is not handled here at all
        if isinstance(batch["num_nodes_in_shape"], torch.Tensor):
            batch["num_nodes_in_shape"] = batch["num_nodes_in_shape"].int().tolist()

        row, col, bond_types = batch["atom_adj"].coo()
        edge_index = torch.stack([row, col], dim=0)
        bond_types = bond_types.long() - 1
        # get class descriptors for embedding
        bond_types = graph_encoder.init_edges_enc(bond_types)

        # transform embedding through MP layers
        node_embeddings = graph_encoder.enc_layers(node_feats, edge_index, bond_types)
        atom_feats = torch.cat((node_embeddings, node_feats), dim=1)

        z_graph_rep = torch.repeat_interleave(z_graph, batch["num_nodes_hgraph"], dim=0)
        motif_feats = node_featurizer.motif_feat_transform(batch["feats_in_motif"])
        hidden_feats = torch.cat((z_graph_rep, motif_feats, batch["shape_node_embeddings"]), dim=1)
        hidden_feats = self.input_to_hidden(hidden_feats)

        input = torch.split(atom_feats[batch["nodes_in_shape"].long()], batch["num_nodes_in_shape"])
        # add positional encoding to input (positional is relative to motif and cant just be computed for all atom_idx)
        assert not torch.any(batch["mult_node_in_shape"] == -1)
        atom_mult = torch.split(batch["mult_node_in_shape"], batch["num_nodes_in_shape"])
        atom_mult = [node_featurizer.atom_mult_emb(am.long() + 1) for am in atom_mult]
        input = [torch.cat((i, am), dim=1) for (i, am) in zip(input, atom_mult)]
        return input, hidden_feats

    def correct_leafs(self, k, pred, batch, max_valence, dataset):
        """
        Given an index `k`, predictions `pred`, a batch of data `batch`, maximum valencies `max_valence`, and a dataset `dataset`,
        this function returns the ID for the output embedding and the charge/bond type if they exist for the corresponding leaf node.

        Args:
        - k (int): The index of the node in the input molecule.
        - pred (torch.Tensor): A tensor containing the predicted bond and atom logits.
        - batch (dict): A dictionary containing information about the batch of data.
        - max_valence (torch.Tensor): A tensor containing the maximum valencies for each node in the input molecule.
        - dataset (CustomMoleculeDataset): A dataset object containing information about the molecules in the dataset.

        Returns:
        - A tuple containing:
            - (int) The ID for the output embedding of the corresponding leaf node.
            - (float or None) The charge of the corresponding leaf node, if it exists.
            - (int or None) The type of bond for the corresponding leaf node, if it exists.

        Note:
        - If the input is padding, the function returns (-2, None, None).
        - If the node has no leaf or if it is already at its maximum valency, the function returns (0, None, None).
        - If the node is a leaf, the function determines the maximum allowed valency based on the atom type and charge,
        and selects the highest-valence bond type that is still possible for the node.
        """
        dev = batch["atom_idx"].device
        if k == -1:
            # if input is padding, return padding
            return (-2, None, None)
        bond_logits, atom_logits = pred[: self.num_edges], pred[self.num_edges :]
        if atom_logits.argmax() == 0:
            return (0, None, None)
        atom_id = atom_logits.argmax().unsqueeze(0) - 1
        atom_charge = torch.Tensor([0]).to(dev)
        degree = (batch["atom_adj"][k] > 0).sum()
        if degree not in [2, 4]:
            return (0, None, None)
        # check if core_mol_atom is at full valence already
        current_valence = batch["atom_adj"].sum(-1)[k]
        if current_valence == max_valence[k]:
            return (0, None, None)
        difference = max_valence[k] - current_valence
        # also check allowed bonds for leaf atom
        max_leaf_valence = get_max_valency(atom_id, atom_charge, dataset.id_to_atom).squeeze()
        difference = torch.min(max_leaf_valence, difference)
        assert difference > 0
        # zero out impossible bonds
        bond_type = bond_logits[: difference.int()].argmax() + 1
        return (atom_id.item() + 1, atom_charge.item(), bond_type.item())

    def forward_inference(self, batch, z_graph, parent_ref):
        dataset = parent_ref.dataset_ref
        # assert not torch.any(batch["shape_classes"][:, 0] == -1)
        dev = z_graph.device
        if isinstance(batch["num_nodes_in_shape"], torch.Tensor):
            batch["num_nodes_in_shape"] = batch["num_nodes_in_shape"].int().tolist()
        # prepare features
        node_feats = parent_ref.node_featurizer(batch)["node_feats"]
        rnn_input, hidden_feats = self.prepare_features(
            batch, z_graph, parent_ref.graph_encoder, parent_ref.node_featurizer, node_feats
        )
        rnn_input = self.input_padding(rnn_input, batch, hidden_feats)
        batch["atom_adj"] = batch["atom_adj"].to_dense()
        max_valence = get_max_valency(batch["atom_idx"], batch["atom_charges"], dataset.id_to_atom)
        nodes_in_shape_split = torch.split(batch["nodes_in_shape"], batch["num_nodes_in_shape"])
        start_tokens = self.output_emb(torch.full((1,), fill_value=-1, device=dev).long() + 2)
        previous_output = start_tokens.repeat(hidden_feats.size(0), 1)
        all_outputs = []
        # hidden_feats = hidden_feats.unsqueeze(0)
        current_input = torch.cat((rnn_input[:, 0, :], previous_output), dim=-1).unsqueeze(1)
        bs, seq_len, dim = current_input.size()
        current_input = self.to_target(current_input.view(-1, dim)).reshape([bs, seq_len, -1])
        for k in range(rnn_input.size(1)):
            # leaf_logits, hidden_feats = self.rec_cell(input=current_input, hx=hidden_feats.contiguous())

            _shape_emb = torch.repeat_interleave(
                hidden_feats.contiguous(), torch.tensor(batch["num_nodes_in_shape"]).to(dev), dim=0
            )
            leaf_logits = transformer_forward(
                self.transformer,
                self.to_memory,
                memory=_shape_emb,
                tgt_input=current_input,
                max_pos=self.max_motif_len,
                num_nodes=torch.full((current_input.size(0),), fill_value=k + 1),
                num_nodes_memory=batch["num_nodes_in_shape"],
                tgt_key_padding_mask=torch.zeros((current_input.size(0), k + 1)).to(dev).bool(),
            )
            leaf_logits = leaf_logits[:, -1]

            leaf_logits = self.leaf_classifier(leaf_logits.squeeze())
            leaf_root_idx = []
            for niss in nodes_in_shape_split:
                if niss.size(0) >= (k + 1):
                    leaf_root_idx.append(niss[k].item())
                else:
                    leaf_root_idx.append(-1)
            assert leaf_logits.size(0) == len(leaf_root_idx)
            output_corrected = [
                self.correct_leafs(lri, ll, batch, max_valence, dataset)
                for (lri, ll) in zip(leaf_root_idx, leaf_logits)
            ]
            output_corrected_ids = torch.tensor([oc[0] for oc in output_corrected]).to(dev)
            all_outputs.append(output_corrected)
            previous_output = self.output_emb(output_corrected_ids.long() + 2)
            if (k + 1) < rnn_input.size(1):
                current_input_new = torch.cat((rnn_input[:, k + 1, :], previous_output), dim=-1).unsqueeze(1)
                bs, seq_len, dim = current_input_new.size()
                current_input_new = self.to_target(current_input_new.view(-1, dim)).reshape([bs, seq_len, -1])
                current_input = torch.cat((current_input, current_input_new), dim=1)
        # batch["atom_adj"] = batch["atom_adj"].to_dense()

        # list transpose and filter out padding
        output_corrected = [[out[j] for out in all_outputs if out[j][0] != -2] for j in range(len(all_outputs[0]))]
        output_corrected = list(chain(*output_corrected))
        has_leaf_node = torch.zeros_like(batch["atom_idx"]).bool().to(dev)
        # iterate through sorted predictions s.t. add and shift index works
        sorted_nodes_in_shape, sorted_idx = batch["nodes_in_shape"].cpu().sort()
        output_corrected = np.array(output_corrected)[sorted_idx.cpu()]
        # average over hypernodes s.t. prediction is unified and order does not matter
        atom_adj = batch["atom_adj"]
        atom_idx = batch["atom_idx"]
        atom_charges = batch["atom_charges"]
        c = 1
        for k, pred in zip(sorted_nodes_in_shape, output_corrected):
            ind = k + c
            # if the node has a leaf already, it was assigned from a different motif iteration
            if has_leaf_node[k]:
                assert (sorted_nodes_in_shape == k).sum() > 1
                continue
            atom_id, atom_charge, bond_type = pred
            # no leaf atom was predicted
            if atom_id == 0:
                continue
            atom_id = atom_id - 1
            atom_id = torch.tensor([atom_id]).to(dev)
            atom_charge = torch.tensor([atom_charge]).to(dev)
            has_leaf_node[k] = True
            m = atom_adj.size(0)
            atom_idx = torch.cat([atom_idx[:ind], atom_id, atom_idx[ind:]])
            atom_charges = torch.cat([atom_charges[:ind], atom_charge, atom_charges[ind:]])
            # extend adjacency with new nodes
            atom_adj = torch.cat([atom_adj[:ind], torch.zeros((1, m)).to(dev), atom_adj[ind:]], dim=0)
            atom_adj = torch.cat([atom_adj[:, :ind], torch.zeros((m + 1, 1)).to(dev), atom_adj[:, ind:]], dim=1)
            # quick sanity check whether we are inserting into the correct column
            assert atom_adj[ind].sum() == atom_adj[:, ind].sum() == 0
            atom_adj[ind, ind - 1] = atom_adj[ind - 1, ind] = bond_type
            c += 1
        # update the nodes per sample with leafs
        for k, hl in enumerate(torch.split(has_leaf_node, batch["num_nodes"])):
            batch["num_nodes"][k] += hl.sum().item()
        batch["atom_idx"] = atom_idx
        batch["atom_charges"] = atom_charges
        batch["atom_adj"] = atom_adj
        batch["leaf_logits"] = leaf_logits
        assert (
            atom_idx.size(0) == atom_charges.size(0) == atom_adj.size(0) == atom_adj.size(1) == sum(batch["num_nodes"])
        )
        assert torch.all(
            batch["atom_adj"].sum(0) <= get_max_valency(batch["atom_idx"], batch["atom_charges"], dataset.id_to_atom)
        )
        return dict(), batch

    def calculate_loss(self, batch, decoder_outputs, dataset):
        # Prepare targets
        atom_target, bond_target = batch["leaf_target"]
        atom_target = atom_target.reshape(-1)
        bond_target = bond_target.reshape(-1)
        atom_exists = bond_target != 0
        bond_target = bond_target[atom_exists]
        atom_target[atom_target == dataset.sequence_tokens["pad_token"]] = self.num_atoms + 1
        bond_target[bond_target == dataset.sequence_tokens["pad_token"]] = self.num_edges + 2
        bond_target = bond_target - 1

        # Prepare outputs
        leaf_logits = decoder_outputs["leaf_logits"]
        bond_logits, atom_logits = leaf_logits[:, : self.num_edges], leaf_logits[:, self.num_edges :]
        bond_logits = bond_logits.reshape(-1, self.num_edges)
        bond_logits = bond_logits[atom_exists]
        weights = calculate_class_weights(atom_target[atom_target != (self.num_atoms + 1)], self.num_atoms)

        atom_ignore = int(self.num_atoms + 1)
        bond_ignore = int(self.num_edges + 1)
        loss_leaf_atoms = CrossEntropyLoss(ignore_index=atom_ignore, weight=weights)(atom_logits, atom_target)
        loss_leaf_bonds = CrossEntropyLoss(ignore_index=bond_ignore)(bond_logits, bond_target)
        loss_leafs = loss_leaf_atoms + loss_leaf_bonds

        # also compute accuracy here s.t. we can track progress
        atom_mask = atom_target != atom_ignore
        atom_acc = calculate_balanced_acc(atom_target[atom_mask], atom_logits[atom_mask])
        bond_mask = bond_target != bond_ignore
        bond_acc = calculate_balanced_acc(bond_target[bond_mask], bond_logits[bond_mask])

        return dict(
            loss_leafs=loss_leafs,
            loss_leaf_atoms=loss_leaf_atoms,
            atom_acc=torch.Tensor([atom_acc]).squeeze(),
            bond_acc=torch.Tensor([bond_acc]).squeeze(),
            loss_leaf_bonds=loss_leaf_bonds,
        )
