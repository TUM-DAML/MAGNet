import importlib
import json
from functools import partial
from pathlib import Path
from typing import Union

from rdkit import Chem

# To be adjusted by user
config = dict(
    SEML_CONFIG_FILE="/path/to/seml/config",
    SOURCE_DIR="path/to/this/repository",
    WB_PROJECT="molecule-generation",
    WB_ENTITY="molgen",
)

SEML_CONFIG_PATH = Path(config["SEML_CONFIG_FILE"])
SOURCE_DIR = Path(config["SOURCE_DIR"])
BASELINE_DIR = SOURCE_DIR / "project_dir"
DATA_DIR = BASELINE_DIR / "data"
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
