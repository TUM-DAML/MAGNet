import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)

from models.magnet.src.data.utils import *
from models.magnet.src.model.load_utils import load_model_from_id
from models.magnet.src.utils import manual_batch_to_device


class LatentDataModule(pl.LightningDataModule):
    """
    Lightning Data Module to handle different datasets & loaders
    """

    def __init__(self, collection, magnet_id, batch_size, dataset, num_workers=8):
        super().__init__()
        self.collection = collection
        self.model_id = magnet_id
        self.dataset = dataset
        self.dl_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": latent_collate,
            "shuffle": True,
        }

    def setup(self, stage=None):
        self.train_ds = LatentDataset(collection=self.collection, model_id=self.model_id, dataset=self.dataset)
        self.val_ds = self.train_ds.magnet_model.dataset_ref
        self.test_ds = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.dl_args)

    def val_dataloader(self):
        # returns only dummy
        return DataLoader(self.train_ds, **self.dl_args)

    def test_dataloader(self):
        return None


class LatentDataset(Dataset):
    def __init__(self, collection, model_id, dataset):
        super().__init__()
        self.magnet_model = load_model_from_id(collection, model_id, dataset=dataset)
        data_loader = self.magnet_model.trainer.datamodule.train_dataloader()
        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

        z_means, z_stds = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader):
                manual_batch_to_device(batch, self.magnet_model.device)
                encoder_outputs = self.magnet_model.encode_graph(batch)
                encoder_outputs = self.magnet_model.latent_module(encoder_outputs)
                z = encoder_outputs["z_graph_dist_params"].detach()
                z_mean, z_std = torch.split(z, z.size(1) // 2, 1)
                z_std = torch.exp(-torch.abs(z_std) / 2)
                z_means.extend([z for z in z_mean.detach().cpu().numpy()])
                z_stds.extend([z for z in z_std.detach().cpu().numpy()])

        self.z_mean = z_means
        self.z_std = z_stds
        print("Gathered all embeddings. Preprocessing done...", flush=True)

    def __len__(self):
        return len(self.z_mean)

    def __getitem__(self, idx):
        return dict(z_mean=self.z_mean[idx], z_std=self.z_std[idx], FM=self.FM)

    def get_magnet_model(self):
        return self.magnet_model


def latent_collate(batches):
    z_mean = torch.tensor(np.stack([batch["z_mean"] for batch in batches]))
    z_std = torch.tensor(np.stack([batch["z_std"] for batch in batches]))
    x1 = z_mean + 0.05 * z_std * torch.randn_like(z_mean)
    x0 = torch.randn_like(x1)
    FM = batches[0]["FM"]
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
    return dict(t=t, xt=xt, ut=ut, x0=x0, x1=x1)
