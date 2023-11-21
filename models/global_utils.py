import importlib
import json
from functools import partial
from pathlib import Path
from typing import Union

from rdkit import Chem


# To be adjusted by user
WB_CONFIG = dict(
    WB_PROJECT="baseline-runs",
    WB_ENTITY="lj-molgen",
)
SEML_CONFIG_PATH = Path("/nfs/homedirs/sommer/.config/seml/mongodb.config")
SOURCE_DIR = Path("/ceph/ssd/staff/sommer/MAGNet/")
BASELINE_DIR = SOURCE_DIR / "project_dir"
DATA_DIR = BASELINE_DIR / "data"
SMILES_DIR = BASELINE_DIR / "smiles_files"
CKPT_DIR = BASELINE_DIR / "model_ckpts"
WB_LOG_DIR = BASELINE_DIR / "wb_logs"

SUPPORTED_MODELS = [
    "JTVAE",
    "PSVAE",
    "SMILESLSTM",
    "HIERVAE",
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
    module = importlib.import_module(f"models.{model_name.lower()}.wrapper_utils")
    return dict(
        train=module.run_training,
        load=module.get_model_func,
        preprocessing=module.run_preprocessing,
    )


def smiles_from_file(file_path: Union[str, Path]):
    with open(file_path, "r") as file:
        smiles = file.readlines()
    smiles = [gt.strip("\n") for gt in smiles]
    assert all([Chem.MolFromSmiles(s) is not None for s in smiles])
    return smiles


def get_model_config(model_name, dataset):
    path = SOURCE_DIR / "models" / model_name.lower() / "model_config.json"
    params = json.load(open(path, "r"))[dataset]
    return params


def handle_tensorflow(model_name: str):
    import tensorflow as tf

    # deactivate Tensorflow when it is not needed as it occupies GPU memory
    if model_name not in TENSORFLOW_MODELS:
        tf.config.experimental.set_visible_devices([], "GPU")
