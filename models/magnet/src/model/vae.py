from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from more_itertools import chunked
from tqdm import tqdm

from baselines.magnet.src.chemutils.rdkit_helpers import pred_to_mols
from baselines.magnet.src.data.mol_module import collate_fn
from baselines.magnet.src.model.atom_featurizers import AtomFeaturizer
from baselines.magnet.src.model.encoders import GraphEncoder
from baselines.magnet.src.model.join_decoder import JoinDecoder
from baselines.magnet.src.model.latent_module import LatentModule
from baselines.magnet.src.model.leaf_decoder import LeafDecoder
from baselines.magnet.src.model.motif_decoder import MotifDecoder
from baselines.magnet.src.model.shape_connectivity_decoder import (
    ShapeConnectivityPredictor,
)
from baselines.magnet.src.model.shape_multiset_decoder import ShapeMultisetPredictor
from baselines.magnet.src.utils import manual_batch_to_device, mol_standardize


class MAGNet(pl.LightningModule):
    def __init__(
        self,
        dim_config: dict,
        feature_sizes: dict,
        lr: float,
        lr_sch_decay: float,
        loss_weights: dict,
        beta_annealing: dict,
        layer_config: dict,
    ):
        """
        Input Arguments
        dim_config: dict
            Dictionary containing all dimensions of the model, e.g. embeddings or hidden dims
            latent_dim: int, dimension of the latent space
            atom_id_dim: int, dimension of the atom type embeddings
            atom_charge_dim: int, dimension of the atom charge embeddings
            shape_id_dim: int, dimension of the shape type embeddings
            atom_multiplicity_dim: int, dimension of the atom multiplicity within shape embeddings
            shape_multiplicity_dim: int, dimension of the shape multiplicity embeddings
            motif_positional_dim: int, dimension of the motif positional embeddings
            motif_seq_positional_dim: int, dimension of the motif positional token embeddings
            motif_feat_dim: int, dimension of the motif feature embeddings
            enc_atom_dim: int, dimension of the atom encoder output
            enc_shapes_dim: int, dimension of the shape encoder output
            enc_joins_dim: int, dimension of the join encoder output
            enc_leafs_dim: int, dimension of the leaf encoder output
            enc_global_dim: int, dimension of the global encoder output
            leaf_rnn_hidden: int, dimension of the leaf decoder RNN hidden state
            shape_rnn_hidden: int, dimension of the shape decoder RNN hidden state
            shape_gnn_dim: int, dimension of the shape embedding GNN hidden state
            max_shape_mult: int, maximum number of shapes of one type in a molecule
            max_atom_mult: int, maximum number of atoms of one type in a shape
        feature_sizes: dict
            Dictionary containing all feature sizes of the dataset, not specified by user
        lr: float
            Learning rate of the optimizer
        lr_sch_decay: float
            Decay rate of the learning rate scheduler
        loss_weights: dict
            Dictionary containing all the weights to aggregate all losses
            shapeset: float, weight of the shape multiset loss
            shapeadj: float, weight of the shape connectivity loss
            motifs: float, weight of the motif loss
            joins: float, weight of the join loss
            leafs: float, weight of the leaf loss
        beta_annealing: dict
            Dictionary containing the beta annealing schedule for the KL term of the VAE
            init: float, initial value of beta
            max: float, maximum value of beta
            start: int, step to start annealing
            every: int, step to increase beta
        layer_config: dict
            Dictionary containing all layer configurations of the model
            num_layers_enc: int, number of layers for the atom encoder
            num_layers_hgraph: int, number of layers for the shape embedding GNN
            num_layers_latent: int, number of layers for the latent MLPs
            num_layers_shape_enc: int, number of layers for the shape encoder
            node_aggregation: str, aggregation function for GNNs
        """
        super(MAGNet, self).__init__()
        self.name = "MAGNet"
        # input feature sizes
        self.feature_sizes = feature_sizes
        # nn dimensions defined by user
        self.dim_config = dim_config
        # decoder / encoder layer configuration
        self.layer_config = layer_config

        # training and loss specifications
        self.lr = lr
        self.lr_sch_decay = lr_sch_decay
        self.loss_weights = loss_weights
        self.beta = beta_annealing["init"]
        self.beta_annealing = beta_annealing

        # initialize modules
        self.node_featurizer = AtomFeaturizer(self)
        self.graph_encoder = GraphEncoder(self)
        self.shapeset_decoder = ShapeMultisetPredictor(self)
        self.shapeadj_decoder = ShapeConnectivityPredictor(self)
        self.shape_to_motif_decoder = MotifDecoder(self)
        self.join_decoder = JoinDecoder(self)
        self.leaf_decoder = LeafDecoder(self)
        self.latent_module = LatentModule(self)
        self.validation_outputs = []

    def _anneal_parameter(self, param_name: str):
        """
        Anneal given parameter based on predefined specifications
        """
        param = getattr(self, param_name)
        config = getattr(self, f"{param_name}_annealing")
        if self.trainer.global_step % config["every"] == 0 and self.trainer.global_step >= config["start"]:
            setattr(self, param_name, min(config["max"], param + config["step"]))

    def set_dataset(self, incoming_dataset=None, incoming_datamodule=None):
        """
        Set dataset for model
        """
        if incoming_dataset is None:
            self.dataset_ref = self.trainer.datamodule.train_ds
            self.datamodule_ref = self.trainer.datamodule
        else:
            self.dataset_ref = incoming_dataset
            self.datamodule_ref = incoming_datamodule

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[object]]:
        """
        Configure optimizer and lr scheduler, is only called by torch lightning
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_sch_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.lr_sch_decay, patience=1)
        schedulers = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "train_loss",
            "frequency": 1,
        }
        return [optimizer], [schedulers]

    def validation_step(self, batch, batch_idx) -> dict:
        """
        Perform forward pass on validation data, is only called by torch lightning
        """
        forward_outputs = self(batch)
        stat_dict = self.collect_all_losses(batch, forward_outputs, training=False)
        stat_dict["z_mean"] = forward_outputs[0]["z_graph_mean"]
        self.validation_outputs.append(stat_dict)

    def on_validation_epoch_end(self):
        """
        Aggregate metrics over validation set and log
        """
        val_outputs = self.validation_outputs
        for key in val_outputs[0].keys():
            if key == "z_mean":
                continue
            res = [batch_result[key].detach().cpu().item() for batch_result in val_outputs]
            self.log("val_" + key, np.array(res).mean())
        # log also the active unit rate
        z_means = torch.cat([batch_result["z_mean"].detach().cpu() for batch_result in val_outputs], dim=0)
        active_units = z_means.var(0) > 0.02  # def from https://arxiv.org/pdf/1706.03643.pdf
        self.log("active_units", active_units.float().mean())
        self.validation_outputs = []

    def training_step(self, batch: dict, batch_idx: int) -> float:
        """
        Training step incl. forward pass and loss calculation, is only called by torch lightning
        """
        if not hasattr(self, "dataset_ref"):
            self.set_dataset()
        forward_outputs = self(batch)
        loss_dict = self.collect_all_losses(batch, forward_outputs, training=True)

        self.log("train_lr", self.optimizers().param_groups[0]["lr"])
        self.log("kl_beta", float(self.beta))

        for loss, loss_value in loss_dict.items():
            self.log("train_" + loss, loss_value, batch_size=len(batch["num_nodes"]))

        self._anneal_parameter("beta")
        return loss_dict["loss"]

    def forward(self, batch: dict) -> Tuple[dict, dict]:
        """
        Full forward pass or the VAE
        """
        encoder_outputs = self.encode_graph(batch)
        encoder_outputs = self.latent_module(encoder_outputs)
        decoder_outputs = self.decode_graph(encoder_outputs["z_graph_decoder"], batch, inference=False)
        return encoder_outputs, decoder_outputs

    def encode_graph(self, batch: dict) -> dict:
        """
        Encode input graph with our multi-level encoder (does not return final latent representation!)
        """
        return self.graph_encoder(batch, self.node_featurizer)

    def decode_graph(self, z_graph: torch.tensor, batch: dict, inference: bool) -> dict:
        """
        Decode graph from "z_graph_decoder" in training or inference mode
        """
        decoder_outputs = dict()
        shapeset_outputs, batch = self.shapeset_decoder(batch, z_graph, inference, self)
        decoder_outputs.update(shapeset_outputs)
        shapeadj_outputs, batch = self.shapeadj_decoder(batch, z_graph, inference, self)
        decoder_outputs.update(shapeadj_outputs)
        motif_outputs, batch = self.shape_to_motif_decoder(batch, z_graph, inference, self)
        decoder_outputs.update(motif_outputs)
        join_outputs, batch = self.join_decoder(batch, z_graph, inference, self)
        decoder_outputs.update(join_outputs)
        leaf_outputs, batch = self.leaf_decoder(batch, z_graph, inference, self)
        decoder_outputs.update(leaf_outputs)
        return batch if inference else decoder_outputs

    def sample_molecules(self, num_samples: int, largest_comp: bool = True) -> List[str]:
        """
        Sample given number of molecules
        """
        sampled_smiles = []
        while len(sampled_smiles) < num_samples:
            embeddings = self.latent_module.sample_gaussian(self.batch_size).to(self.device)
            smiles = self.decode_from_latent_mean(embeddings, largest_comp=largest_comp)
            sampled_smiles.extend(smiles)
        sampled_smiles = sampled_smiles[:num_samples]
        return sampled_smiles

    def batch_from_smiles(self, input: List[str]) -> dict:
        """
        Create batch from input smiles (similar to dataloader, but this is used at inference time for unseen data)
        """
        samples = [self.dataset_ref._getitem(None, smiles=s) for s in input]
        batch = collate_fn(samples)
        manual_batch_to_device(batch, self.device)
        return batch

    def reconstruct_from_smiles(
        self, input_smiles: List[str], sample_latent: bool = False, largest_comp: bool = True
    ) -> List[str]:
        """
        Performs full reconstruction from given list of smiles
        """
        reconstructed_smiles = []
        for inp in tqdm(chunked(input_smiles, self.batch_size)):
            batch = self.batch_from_smiles(inp)
            z_graph = self.encode_to_latent_mean(batch, sample=sample_latent)
            smiles = self.decode_from_latent_mean(z_graph, to_smiles=True, largest_comp=largest_comp)
            reconstructed_smiles.extend(smiles)
        return reconstructed_smiles

    def encode_to_latent_mean(self, batch: dict, sample: bool = False) -> torch.tensor:
        """
        Encodes input batch to true latent mean
        """
        encoder_outputs = self.encode_graph(batch)
        latent_outputs = self.latent_module(encoder_outputs, sample=sample)
        return latent_outputs["z_graph_rsampled"]

    def decode_from_latent_mean(
        self, z_graph: torch.tensor, to_smiles: bool = True, largest_comp: bool = True
    ) -> Union[List[str], dict]:
        """
        Decodes smiles from true latent mean
        """
        embeddings = self.latent_module.decode_latent(z_graph)
        sampled_mols = self.decode_graph(embeddings, None, inference=True)
        if to_smiles:
            id_to_atom_map = self.dataset_ref.id_to_atom
            mols = pred_to_mols(sampled_mols, id_to_atom_map)
            smiles = [mol_standardize(m, largest_comp=largest_comp) for m in mols]
            return smiles
        return sampled_mols

    def collect_all_losses(self, batch: dict, forward_outputs: dict, training: bool) -> dict:
        """
        Collect losses of individual modules and aggregate them to overall loss
        """
        dataset = self.dataset_ref
        encoder_outputs, decoder_outputs = forward_outputs

        # collect all losses
        loss_dict = self.shapeset_decoder.calculate_loss(batch, decoder_outputs, dataset)
        loss_dict.update(self.shapeadj_decoder.calculate_loss(batch, decoder_outputs, dataset))
        loss_dict.update(self.shape_to_motif_decoder.calculate_loss(batch, decoder_outputs, dataset))
        loss_dict.update(self.join_decoder.calculate_loss(batch, decoder_outputs, dataset))
        loss_dict.update(self.leaf_decoder.calculate_loss(batch, decoder_outputs, dataset))
        loss_dict.update(self.latent_module.calculate_loss(encoder_outputs))

        # aggregate loss with weighting
        loss_overall = 0
        for key, value in self.loss_weights.items():
            loss_overall += (value if training else 1) * loss_dict["loss_" + key]
        loss_overall += (self.beta if training else 1) * loss_dict["kl_loss"]
        loss_dict["loss"] = loss_overall
        return loss_dict
