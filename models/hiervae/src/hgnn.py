import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.hiervae.src.decoder import HierMPNDecoder
from models.hiervae.src.encoder import HierMPNEncoder
from models.hiervae.src.nnutils import *
from models.hiervae.src.vocab import common_atom_vocab


class HierVAE(pl.LightningModule):
    def __init__(
        self,
        vocab=None,
        atom_vocab=common_atom_vocab,
        rnn_type="LSTM",
        embed_size=250,
        hidden_size=250,
        depthT=15,
        depthG=15,
        dropout=0.0,
        latent_size=32,
        diterT=1,
        diterG=3,
        beta=0,
        kl_warmup=10000,
        kl_anneal=2000,
        step_beta=0.0001,
        max_beta=1.0,
    ):
        super(HierVAE, self).__init__()
        self.name = "HierVAE"
        self.encoder = HierMPNEncoder(
            vocab,
            atom_vocab,
            rnn_type,
            embed_size,
            hidden_size,
            depthT,
            depthG,
            dropout,
        )
        self.decoder = HierMPNDecoder(
            vocab,
            atom_vocab,
            rnn_type,
            embed_size,
            hidden_size,
            latent_size,
            diterT,
            diterG,
            dropout,
        )
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = latent_size

        self.R_mean = nn.Linear(hidden_size, latent_size)
        self.R_var = nn.Linear(hidden_size, latent_size)
        self.beta = beta
        self.kl_anneal = kl_anneal
        self.kl_warmup = kl_warmup
        self.step_beta = step_beta
        self.max_beta = max_beta

        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        return [optimizer], [scheduler]

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

    def reconstruct(self, batch):
        graphs, tensors, _ = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def training_step(self, batch, batch_idx, perturb_z=True):
        graphs, tensors, orders = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        root_vecs, _, _, _ = self.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        loss = loss + self.beta * kl_div
        if self.trainer.global_step % self.kl_anneal == 0 and self.trainer.global_step >= self.kl_warmup:
            self.beta = min(self.max_beta, self.beta + self.step_beta)

        self.log("train_loss", loss.detach())
        self.log("word_accuracy", wacc.detach())
        self.log("topo_accuracy", tacc.detach())
        self.log("iword_accuracy", iacc.detach())
        self.log("assm_accuracy", sacc.detach())
        self.log("kl_loss", kl_div.detach())
        self.log("current_beta", torch.Tensor([self.beta]).float())
        self.log("current_lr", self.optimizers().param_groups[0]["lr"])
        return loss


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


class HierVGNN(pl.LightningModule):
    def __init__(
        self,
        vocab,
        atom_vocab=common_atom_vocab,
        rnn_type="LSTM",
        embed_size=270,
        hidden_size=270,
        depthT=20,
        depthG=20,
        dropout=0.0,
        latent_size=4,
        diterT=1,
        diterG=3,
        beta=0.3,
    ):
        super(HierVGNN, self).__init__()
        self.latent_size = latent_size
        self.name = "HierPROP"
        self.encoder = HierMPNEncoder(
            vocab,
            atom_vocab,
            rnn_type,
            embed_size,
            hidden_size,
            depthT,
            depthG,
            dropout,
        )
        self.decoder = HierMPNDecoder(
            vocab,
            atom_vocab,
            rnn_type,
            embed_size,
            hidden_size,
            hidden_size,
            diterT,
            diterG,
            dropout,
            attention=True,
        )
        self.encoder.tie_embedding(self.decoder.hmpn)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

        self.W_tree = nn.Sequential(nn.Linear(hidden_size + latent_size, hidden_size), nn.ReLU())
        self.W_graph = nn.Sequential(nn.Linear(hidden_size + latent_size, hidden_size), nn.ReLU())

        self.beta = beta
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        return [optimizer], [scheduler]

    def encode(self, tensors):
        tree_tensors, graph_tensors = tensors
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        tree_vecs = stack_pad_tensor([tree_vecs[st : st + le] for st, le in tree_tensors[-1]])
        graph_vecs = stack_pad_tensor([graph_vecs[st : st + le] for st, le in graph_tensors[-1]])
        return root_vecs, tree_vecs, graph_vecs

    def translate(self, tensors, num_decode, enum_root, greedy=True):
        tensors = make_cuda(tensors)
        root_vecs, tree_vecs, graph_vecs = self.encode(tensors)
        all_smiles = []
        if enum_root:
            repeat = num_decode // len(root_vecs)
            modulo = num_decode % len(root_vecs)
            root_vecs = torch.cat([root_vecs] * repeat + [root_vecs[:modulo]], dim=0)
            tree_vecs = torch.cat([tree_vecs] * repeat + [tree_vecs[:modulo]], dim=0)
            graph_vecs = torch.cat([graph_vecs] * repeat + [graph_vecs[:modulo]], dim=0)

        batch_size = len(root_vecs)
        z_tree = torch.randn(batch_size, 1, self.latent_size).expand(-1, tree_vecs.size(1), -1).cuda()
        z_graph = torch.randn(batch_size, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
        z_tree_vecs = self.W_tree(torch.cat([tree_vecs, z_tree], dim=-1))
        z_graph_vecs = self.W_graph(torch.cat([graph_vecs, z_graph], dim=-1))
        return self.decoder.decode((root_vecs, z_tree_vecs, z_graph_vecs), greedy=greedy)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def training_step(self, batch, batch_idx):
        loss, kl_div, wacc, iacc, tacc, sacc = self(*batch)

        # Learning Rate Scheduler
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch.step()

        self.log("train_loss", loss.detach())
        self.log("word_accuracy", wacc.detach())
        self.log("topo_accuracy", tacc.detach())
        self.log("iword_accuracy", iacc.detach())
        self.log("assm_accuracy", sacc.detach())
        self.log("kl_loss", kl_div)
        self.log("current_beta", torch.Tensor([self.beta]).float())
        self.log("current_lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_root_vecs, x_tree_vecs, x_graph_vecs = self.encode(x_tensors)
        _, y_tree_vecs, y_graph_vecs = self.encode(y_tensors)

        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_graph_vecs, graph_kl = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + graph_kl

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_tree_vecs = self.W_tree(torch.cat([x_tree_vecs, diff_tree_vecs], dim=-1))
        x_graph_vecs = self.W_graph(torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1))

        loss, wacc, iacc, tacc, sacc = self.decoder(
            (x_root_vecs, x_tree_vecs, x_graph_vecs), y_graphs, y_tensors, y_orders
        )
        return loss + self.beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc
