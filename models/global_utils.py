import importlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Union

import yaml
from rdkit import Chem

# To be adjusted by user
config = dict(
    SEML_CONFIG_FILE="/nfs/homedirs/sommer/.config/seml/mongodb.config",
    BASELINE_DIR="/ceph/ssd/staff/sommer/molgen/baselines",
    SOURCE_DIR="/nfs/homedirs/sommer/baselines-molgen/",
    WB_PROJECT="baseline-runs",
    WB_ENTITY="lj-molgen",
)

SEML_CONFIG_PATH = Path(config["SEML_CONFIG_FILE"])
BASELINE_DIR = Path(config["BASELINE_DIR"])
DATA_DIR = BASELINE_DIR / "data"
SOURCE_DIR = Path(config["SOURCE_DIR"])
SUPPORTED_MODELS = [
    "JTVAE",
    "PSVAE",
    "SMILESLSTM",
    "HIERVAE",
    "MICAM",
    "GCPN",
    "GRAPHAF",
    "MOLER",
    "CHARVAE",
    "MAGNET",
]
SUPPORTED_DATA = ["zinc"]
TENSORFLOW_MODELS = ["CHARVAE", "MOLER"]


def get_all_model_funcs(model_name: str) -> dict:
    handle_tensorflow(model_name)
    module = importlib.import_module(f"baselines.{model_name.lower()}.wrapper_utils")
    return dict(
        train=partial(module.run_training, config=config),
        load=partial(module.get_model_func, config=config),
    )


def smiles_from_file(file_path: Union[str, Path]):
    with open(file_path, "r") as file:
        smiles = file.readlines()
    smiles = [gt.strip("\n") for gt in smiles]
    assert all([Chem.MolFromSmiles(s) is not None for s in smiles])
    return smiles


def get_model_config(config, model_name, dataset):
    path = Path(config["SOURCE_DIR"]) / "baselines" / model_name.lower() / "model_config.json"
    params = json.load(open(path, "r"))[dataset]
    return params


def handle_tensorflow(model_name: str):
    import tensorflow as tf

    # deactivate Tensorflow when it is not needed as it occupies GPU memory
    if model_name not in TENSORFLOW_MODELS:
        tf.config.experimental.set_visible_devices([], "GPU")
