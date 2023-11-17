import json
import time
from typing import List

import numpy as np
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

# from moses import get_all_metrics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from baselines.global_utils import BASELINE_DIR, smiles_from_file


def calculate_all_sampling_metrics(smiles, dataset):
    if dataset == "zinc":
        train_path = BASELINE_DIR / "smiles_files" / dataset / "train.txt"
        test_path = BASELINE_DIR / "smiles_files" / dataset / "test.txt"
    else:
        raise NotImplementedError()
    # metrics can only be computed with > 10000 smiles
    assert len(smiles) >= 10000

    results = dict(
        Guacamol_Metrics=calculate_guacamol_benchmark(train_path, smiles),
        Moses_Metrics=get_all_metrics(smiles, test=smiles_from_file(test_path)),
    )
    return results


class DummyGenerator(DistributionMatchingGenerator):
    def __init__(self, smiles):
        self.smiles = smiles

    def generate(self, number_samples: int):
        np.random.shuffle(self.smiles)
        return self.smiles[:number_samples]


def calculate_guacamol_benchmark(dataset_path, generated_smiles):
    model_generator = DummyGenerator(generated_smiles)
    json_path = BASELINE_DIR / "wb_logs" / "guacamol_jsons" / (str(time.time()) + ".json")
    assess_distribution_learning(
        model_generator,
        chembl_training_file=dataset_path,
        json_output_file=json_path,
        benchmark_version="v2",
    )
    out_results = dict()
    with open(json_path) as json_file:
        guacamol_results = json.load(json_file)
        for subdict in guacamol_results["results"]:
            out_results[subdict["benchmark_name"]] = subdict["score"]
    return out_results


def calculate_tanimoto(smiles1, smiles2):
    # Convert the SMILES strings to RDKit molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Calculate the Morgan fingerprints for each molecule
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)

    # Calculate the Tanimoto similarity between the fingerprints
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    # Calculate the Tanimoto distance as 1 - similarity
    distance = 1 - similarity

    return distance


def compare_smiles(smiles1, smiles2):
    # Convert the SMILES strings to RDKit molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Remove stereochemistry information from the molecules
    Chem.RemoveStereochemistry(mol1)
    Chem.RemoveStereochemistry(mol2)

    # Canonicalize the molecules
    can_smiles1 = Chem.MolToSmiles(mol1, isomericSmiles=False, canonical=True)
    can_smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=False, canonical=True)
    can_smiles1 = Chem.CanonSmiles(can_smiles1)
    can_smiles2 = Chem.CanonSmiles(can_smiles2)

    # Compare the canonical SMILES strings for an exact match
    return can_smiles1 == can_smiles2
