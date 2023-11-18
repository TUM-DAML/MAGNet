import pickle
from typing import Union

import torch

from models.magnet.src.chemutils.constants import INFERENCE_HPS
from models.magnet.src.data.mol_module import MolDataModule
from models.magnet.src.model.flow_vae import FlowMAGNet
from models.magnet.src.model.vae import MAGNet


def load_model_from_id(
    data_dir: str,
    collection: str,
    run_id: str,
    dataset: str = "zinc",
    seed_model: int = 0,
    model_class: object = MAGNet,
    load_config: dict = dict(),
    return_config: bool = False,
) -> Union[MAGNet, FlowMAGNet]:
    """
    Load model from collection / id and set inference parameters
    """
    torch.manual_seed(seed_model)

    # create dummy datamodule, this is only for things like id-to-shape maps and not to utilize the DL
    dm = MolDataModule(dataset, data_dir, batch_size=1, num_workers=6, shuffle=False)
    dm.setup()

    # load model config from file
    checkpoint_dir = data_dir / "wb_logs" / collection / run_id / "checkpoints"
    with open(checkpoint_dir / "load_config.pkl", "rb") as file:
        config = pickle.load(file)

    # Load model weights
    state_dict = torch.load(checkpoint_dir / "last.ckpt")["state_dict"]

    for key in load_config.keys():
        if key in config.keys():
            config.pop(key)

    # ensure correct loading of FlowMAGNet, i.e. don't overwrite trained modules
    load_sepearate_modules = False
    if model_class == FlowMAGNet:
        if any("flow_model" in key for key in state_dict.keys()):
            print("Found Flow Modules in state dict, loading full FlowMAGNet model.")
            load_config["load_flow_modules"] = True
        else:
            print("Loading MAGNet model, initializing new Flow Modules!")
            load_config["load_flow_modules"] = False
            load_sepearate_modules = True

    model = model_class(feature_sizes=dm.feature_sizes, **config, **load_config)
    model.load_state_dict(state_dict)

    if load_sepearate_modules:
        print("Initializing Flow Modules...")
        model.initialize_flow_modules()

    model.cuda()
    model.eval()
    model.set_dataset(incoming_dataset=dm.val_ds, incoming_datamodule=dm)

    # set inference parameters according to dataset default
    sampling_params = INFERENCE_HPS[dataset.lower()]
    for key, value in sampling_params.items():
        setattr(model, key, value)
    if return_config:
        config.update(load_config)
        return model, config
    return model
