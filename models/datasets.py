from models.global_utils import SMILES_DIR
import os
import os
import requests
from zipfile import ZipFile
import tarfile
from itertools import chain

DATA_URLS = dict(zinc="https://figshare.com/ndownloader/files/43192467?private_link=e2d9afff8ff6885fb59a")


def check_dataset_exists(dataset_name):
    dataset_name = dataset_name.lower()
    if not (SMILES_DIR / dataset_name).exists():
        print("Dataset not found, downloading now:", dataset_name)
        download_dataset(dataset_name)
    else:
        print("Found dataset in data directories!")


def download_dataset(dataset_name):
    smiles_directory = SMILES_DIR / dataset_name
    url = DATA_URLS[dataset_name]

    os.makedirs(smiles_directory, exist_ok=True)
    file_name = os.path.join(smiles_directory, url.split("/")[-1])

    response = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(response.content)

    with tarfile.open(file_name, 'r:gz') as tar_ref:
        tar_ref.extractall(smiles_directory)
    os.remove(file_name)
    print("Dataset Download successful.")

