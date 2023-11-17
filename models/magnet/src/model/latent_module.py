import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torch_geometric.nn import MLP

from baselines.magnet.src.utils import NUM_STABLE


class LatentModule(torch.nn.Module):
    def __init__(self, parent_ref):
        super().__init__()
        num_layers = parent_ref.layer_config["num_layers_latent"]
        dim_config = parent_ref.dim_config
        input_dim = (
            dim_config["enc_atom_dim"]
            + dim_config["enc_shapes_dim"]
            + dim_config["enc_joins_dim"]
            + dim_config["enc_leafs_dim"]
            + dim_config["enc_global_dim"]
        )
        self.latent_dim = dim_config["latent_dim"]
        self.num_layers = num_layers
        layers_encoder = [input_dim * 2] + [self.latent_dim * 2] * num_layers
        # ATTENTION: Since the RNN modules rely on the latent_dim entry, we do not decode back to the input_dim
        layers_decoder = [self.latent_dim] * (num_layers + 1)
        if num_layers > 0:
            self.encoder = MLP(layers_encoder)
            self.decoder = MLP(layers_decoder)
        self.add_std = dim_config["add_std"]

    def forward(self, encoder_outputs, sample=True):
        input_embedding = encoder_outputs["z_graph_encoder"]
        latent = self.encode_latent(input_embedding)
        encoder_outputs["z_graph_dist_params"] = latent
        encoder_outputs["z_graph_mean"] = self.variational_sample(latent, False)
        latent = self.variational_sample(latent, sample)
        encoder_outputs["z_graph_rsampled"] = latent
        output_embedding = self.decode_latent(latent)
        encoder_outputs["z_graph_decoder"] = output_embedding
        return encoder_outputs

    def sample_gaussian(self, num_mols):
        embeddings = torch.randn(num_mols, self.latent_dim)
        return embeddings

    def decode_latent(self, embedding):
        if self.num_layers > 0:
            return self.decoder(embedding)
        else:
            return embedding

    def encode_latent(self, embedding):
        if self.num_layers > 0:
            return self.encoder(embedding)
        else:
            return embedding

    def variational_sample(self, z, sample=True):
        z_mean, z_var = torch.split(z, z.size(1) // 2, 1)
        z_std = torch.exp(-torch.abs(z_var) / 2) + self.add_std
        latent_dist = Normal(z_mean, (z_std + NUM_STABLE))
        if not sample:
            return latent_dist.mean
        return latent_dist.rsample()

    def calculate_loss(self, encoder_outputs):
        """
        Calculates KL Divergence of approximate posterior to normal prior
        """
        z = encoder_outputs["z_graph_dist_params"]
        z_mean, z_var = torch.split(z, z.size(1) // 2, 1)
        # Prior
        p_z = Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        # Approximate posterior
        q_z = Normal(z_mean, torch.exp(-torch.abs(z_var) / 2) + NUM_STABLE)
        kl_loss = kl(q_z, p_z)
        return dict(kl_loss=kl_loss.mean())
