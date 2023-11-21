import sys
import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from rdkit import RDLogger


class InferenceBase(ABC):
    def __init__(self, model, seed: int, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self.seed = seed
        self.start_inference_server()

        # Interface for Molecule Swarm Optimization
        self.seq_to_emb = self.encode
        self.emb_to_seq = self.decode

    """
    Definition of decoding interfaces for baseline models
    """

    @abstractmethod
    def random_mol(self, num_samples: int) -> List[str]:
        """
        Sample molecules from the prior distribution of the model
        """
        pass

    @abstractmethod
    def encode(self, smiles: List[str]) -> np.array:
        """
        Encode a list of SMILES strings into latent vectors
        """
        pass

    @abstractmethod
    def decode(self, latents: np.array) -> List[str]:
        """
        Decode a list of latent vectors into SMILES strings
        """
        pass

    def valid_check(self, smiles: str) -> bool:
        """
        Check if a molecule is valid according to model's definition
        """
        return True

    """
    Benchmarking & Inference Functionality
    """

    def encode_molecules(self, smiles: List[str]) -> np.array:
        """
        Encode a list of SMILES strings into latent vectors
        """
        latent_vectors = []
        for i in range(0, len(smiles), self.batch_size):
            latent_vectors.append(self.encode(smiles[i : i + self.batch_size]))
        return np.concatenate(latent_vectors)

    def sample_molecules(self, num_samples: int, valid_check=True) -> List[str]:
        """
        Sample a given number of molecules from the model
        """
        output_smiles = []
        while len(output_smiles) < num_samples:
            num_samples_current = min(num_samples - len(output_smiles), self.batch_size)
            sampled_smiles = self.random_mol(num_samples_current)
            if valid_check:
                output_smiles.extend([o for o in sampled_smiles if self.valid_check(o)])
            else:
                output_smiles.extend(sampled_smiles)
        return output_smiles

    def reconstruct_molecules(self, ground_truth_smiles: List[str]) -> List[str]:
        """
        Perform reconstruction on a given set of ground truth smiles
        """
        sample_counter = 0
        output_smiles = []
        while sample_counter < len(ground_truth_smiles):
            num_samples_current = min(len(ground_truth_smiles) - sample_counter, self.batch_size)
            current_gt_smiles = ground_truth_smiles[sample_counter : sample_counter + num_samples_current]
            latent_vectors = self.encode(current_gt_smiles)
            assert len(latent_vectors.shape) > 1
            reconstructed_smiles = self.decode(latent_vectors)
            output_smiles.extend(reconstructed_smiles)
            sample_counter += num_samples_current
        return output_smiles

    def interpolate_between_molecules(self, smiles_pairs: List[List[str]], num_interpolations: int) -> List[str]:
        """
        Perform interpolation between two given sets of ground truth smiles
        """
        all_interpolations = []
        for sp in smiles_pairs:
            embeddings = self.encode(sp)
            # interpolate between embeddings
            embeddings_interpol = []
            for i in range(num_interpolations):
                embeddings_interpol.append(
                    embeddings[0] + (embeddings[1] - embeddings[0]) * i / (num_interpolations - 1)
                )
            embeddings_interpol = np.stack(embeddings_interpol)
            # decode interpolated embeddings
            smiles_list = self.decode(embeddings_interpol)
            smiles_list = [sp[0]] + smiles_list + [sp[1]]
            all_interpolations.append(smiles_list)
        return all_interpolations

    """
    Maintenance of Tensorflow sessions and random seeds
    """

    def start_inference_server(self):
        """
        Set random seed for generation and manage tensorflow session if neccessary
        """
        # TODO: tf, numpy
        torch.manual_seed(self.seed)
        self.enter_session()
        RDLogger.DisableLog("rdApp.error")
        warnings.filterwarnings("ignore")

    def end_inference_server(self):
        """
        Handle sessions and do post-inference cleanup
        """
        self.exit_session()
        # seml has a problem when tensorflow visible devices are altered
        del sys.modules["tensorflow"]
        # TODO: enable logging

    def enter_session(self):
        """
        Start tensorflow session if neccessary
        """
        if isinstance(self.model, tuple):
            self.model = self.model[0](self.model[1])
            self.model.__enter__()
        return self.model

    def exit_session(self):
        """
        End tensorflow session if neccessary
        """
        if isinstance(self.model, tuple):
            self.model.__exit__(None, None, None)
