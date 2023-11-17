from typing import List, Tuple, Union

import torch
from torchdyn.core import NeuralODE

from baselines.magnet.src.chemutils.constants import FLOW_NODE_PARAMS, FLOW_TRAIN_HPS
from baselines.magnet.src.model.vae import MAGNet


class TimeMLP(torch.nn.Module):
    def __init__(self, dim, w=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, 2 * w),
            torch.nn.SELU(),
            torch.nn.Linear(2 * w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim),
        )

    def forward(self, x):
        return self.net(x)


class FlowMAGNet(MAGNet):
    def __init__(self, patience, flow_dim_config, sample_config, load_flow_modules=True, **kwargs):
        # this initialization gives us access to the MAGNet class through self
        super(FlowMAGNet, self).__init__(**kwargs)
        self.patience = patience
        self.sampled_molecules = []
        # freeze all parameters of MAGNet base model
        for param in self.parameters():
            param.requires_grad = False
        self.flow_dim_config = flow_dim_config
        self.sample_config = sample_config
        if load_flow_modules:
            self.initialize_flow_modules()

    def initialize_flow_modules(self):
        self.flow_model = TimeMLP(self.latent_module.latent_dim, self.flow_dim_config["hidden_dim"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_sch_decay, verbose=True, patience=self.patience
        )
        schedulers = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "train_loss",
            "frequency": 1,
        }
        return [optimizer], [schedulers]

    def training_step(self, batch):
        # unpack inputs
        t = batch["t"]
        xt = batch["xt"]
        ut = batch["ut"]

        # flow forward pass
        vt = self.flow_model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        # log training loss and current learning rate
        self.log("train_loss", loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_outputs.append(dict())

    def encode_to_latent_mean(self, batch: dict, sample: bool = False) -> torch.tensor:
        z_graph = super(FlowMAGNet, self).encode_to_latent_mean(batch=batch, sample=sample)
        z_graph = self.flow_backward_mapping(z_graph)
        return z_graph

    def decode_from_latent_mean(
        self, z_graph: torch.tensor, to_smiles: bool = True, largest_comp: bool = True
    ) -> Union[List[str], dict]:
        embeddings = self.flow_forward_mapping(z_graph)
        return super(FlowMAGNet, self).decode_from_latent_mean(
            embeddings, to_smiles=to_smiles, largest_comp=largest_comp
        )

    def flow_backward_mapping(self, initial_state: torch.tensor) -> torch.tensor:
        return self.get_trajectory(initial_state, forward=False)

    def flow_forward_mapping(self, initial_state: torch.tensor) -> torch.tensor:
        return self.get_trajectory(initial_state, forward=True)

    def get_trajectory(self, initial_state: torch.tensor, forward=True) -> torch.tensor:
        if forward:
            t_span = torch.linspace(0, 1, self.sample_config["num_steps"])
        else:
            t_span = torch.linspace(1, 0, self.sample_config["num_steps"])

        node = NeuralODE(
            torchdyn_wrapper(self.flow_model),
            solver=self.sample_config["solver"],
            sensitivity=self.sample_config["sensitivity"],
            atol=self.sample_config["atol"],
            rtol=self.sample_config["rtol"],
        )
        node.to(self.device)
        traj = node.trajectory(
            initial_state,
            t_span=t_span.to(self.device),
        )
        return traj[-1]


class torchdyn_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, args=None):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def get_flow_training_args(input_args):
    # set Neural ODE parameters
    input_args["sample_config"] = FLOW_NODE_PARAMS
    print("Setting Neural ODE parameters: ", input_args["sample_config"])
    # set other default HPs we don't want to specify
    input_args.update(FLOW_TRAIN_HPS)
    print("Setting default Flow Matching hyperparameters: ", FLOW_TRAIN_HPS)
    # calculate number of evaluations
    input_args["val_n_epochs"] = input_args["epochs"] // (input_args["val_n_times"] + 3)
    print("Evaluation every {} epochs".format(input_args["val_n_epochs"]))
    return input_args
