from copy import deepcopy

import networkx as nx
import torch
import torch_sparse
from torch.nn import GRU, CrossEntropyLoss, Embedding
from torch_geometric.nn import MLP
from torch_sparse import SparseTensor

from baselines.magnet.src.chemutils.motif_helpers import (
    atom_counts_to_multiset,
    atom_multiset_to_counts,
    get_atom_pos_encoding,
)
from baselines.magnet.src.chemutils.rdkit_helpers import extract_single_graphs
from baselines.magnet.src.model.transformer_utils import transformer_forward
from baselines.magnet.src.utils import SMALL_INT, calculate_class_weights


class ShapeMultisetPredictor(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        latent_dim = parent_ref.dim_config["latent_dim"]
        self.num_shapes = parent_ref.feature_sizes["num_shapes"]
        self.rnn_hidden_dim = 2 * latent_dim
        self.to_shape = MLP([self.rnn_hidden_dim, self.num_shapes + 3])
        self.rnn_embedding = Embedding(self.num_shapes + 3, self.rnn_hidden_dim)

        _transformer_layers = torch.nn.TransformerDecoderLayer(
            self.rnn_hidden_dim, nhead=8, dim_feedforward=256, batch_first=True
        )
        self.transformer = torch.nn.TransformerDecoder(_transformer_layers, num_layers=2)
        self.to_memory = MLP([latent_dim, self.rnn_hidden_dim])

    def forward(self, batch, z_graph, sample, parent_ref):
        if sample:
            shapeset_outputs, batch = self.forward_inference(z_graph, parent_ref)
        else:
            shapeset_outputs = self.forward_train(batch, z_graph)
        return shapeset_outputs, batch

    def forward_train(self, batch, z_graph):
        rnn_input = self.rnn_embedding((batch["shape_nodes_seq"] + 3))[:, :-1]
        tgt_key_padding_mask = (batch["shape_nodes_seq"] == -3)[:, :-1]

        transformer_output = transformer_forward(
            transformer_nn=self.transformer,
            memory_nn=self.to_memory,
            num_nodes=batch["num_nodes_in_shape"],
            memory=z_graph,
            tgt_input=rnn_input,
            tgt_key_padding_mask=tgt_key_padding_mask,
            max_pos=1,
        )

        shape_logits = self.to_shape(transformer_output)
        return dict(shape_logits=shape_logits)

    def forward_inference(self, z_graph, parent_ref):
        dataset_ref = parent_ref.dataset_ref
        bs = z_graph.size(0)
        device = z_graph.device
        stop_token = dataset_ref.sequence_tokens["end_token"] + 3
        stop_token = torch.full((1,), fill_value=stop_token, device=device, dtype=torch.long)
        input_token = dataset_ref.sequence_tokens["start_token"] + 3
        input_token = torch.full((bs, 1), fill_value=input_token, device=device, dtype=torch.long)
        multiset_counts = torch.zeros((bs, self.num_shapes), device=device)
        stopped = torch.zeros((bs)).bool().to(device)
        # rnn_hidden = self.to_rnn(z_graph.unsqueeze(0))
        shape_sizes = torch.Tensor(list(dataset_ref.shape_to_size.values())).to(device)
        current_size = torch.zeros((bs), device=device)
        while True:
            input_embedding = self.rnn_embedding(input_token)
            prediction = transformer_forward(
                transformer_nn=self.transformer,
                memory_nn=self.to_memory,
                num_nodes=None,
                memory=z_graph,
                tgt_input=input_embedding,
                tgt_key_padding_mask=torch.zeros((input_embedding.size(0), input_embedding.size(1)))
                .to(device)
                .bool(),
                max_pos=1,
            )
            prediction = prediction[:, -1]
            shape_logits = self.to_shape(prediction)
            input_token_new = torch.argmax(shape_logits, dim=-1).squeeze(-1)
            # true when sequence *was stopped at some point*
            stopped = torch.logical_or(input_token_new == stop_token.squeeze(), stopped)
            stop_too_soon = torch.logical_and(stopped, current_size < parent_ref.min_size)
            if torch.any(stop_too_soon):
                fix_idx = stop_too_soon.nonzero().squeeze(1)
                for idx in fix_idx:
                    shape_logits[idx, 1] = SMALL_INT
                    stopped[idx] = False
                input_token_new = torch.argmax(shape_logits, dim=-1).squeeze(-1)

            if input_token_new.dim() == 0:
                input_token_new = input_token_new.unsqueeze(0)

            too_large = current_size >= parent_ref.max_size
            ok_mask = torch.logical_and(~stopped, ~too_large)
            multiset_counts[~stopped, input_token_new[~stopped] - 3] += 1
            # true when generated molecule becomes too large in terms of atoms
            input_token_new = input_token_new.view(-1, 1)
            if not torch.any(ok_mask):
                break
            current_size = (shape_sizes * multiset_counts).sum(1)
            input_token = torch.cat([input_token, input_token_new], dim=1)

        multiset_counts = torch.clamp(multiset_counts, max=parent_ref.dim_config["max_shape_mult"])
        shape_num_nodes = multiset_counts.sum(-1).int()
        shape_idx = atom_counts_to_multiset(multiset_counts.int()).to(device).long()
        shape_idx_split = torch.split(shape_idx, shape_num_nodes.tolist())
        shape_multiplicity = (
            torch.cat([get_atom_pos_encoding(si, torch.ones_like(si)) for si in shape_idx_split], dim=-1)
            .to(device)
            .long()
        )
        return dict(), dict(
            shape_node_idx=shape_idx, shape_node_mult=shape_multiplicity, num_nodes_hgraph=shape_num_nodes
        )

    def calculate_loss(self, batch, decoder_outputs, dataset):
        output = decoder_outputs["shape_logits"]
        target = batch["shape_nodes_seq"] + 3
        # Start from first shape
        target = target[:, 1:].reshape(-1)
        dim = output.size(2)
        padding_token = dataset.sequence_tokens["pad_token"]
        loss_set = CrossEntropyLoss(ignore_index=padding_token + 3)(output.view(-1, dim), target)
        return dict(loss_shapeset=loss_set)
