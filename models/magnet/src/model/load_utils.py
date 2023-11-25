import pickle
from typing import Union

import torch

from models.global_utils import WB_LOG_DIR
from models.magnet.src.data.mol_module import MolDataModule
from models.magnet.src.model.flow_vae import FlowMAGNet
from models.magnet.src.model.vae import MAGNet


def load_model_from_id(
    collection: str,
    run_id: str,
    dataset: str = "zinc",
    seed_model: int = 0,
    model_class: object = MAGNet,
    load_config: dict = dict(),
    return_config: bool = False,
    sampling_params: dict = dict(sample_threshold=0.5, batch_size=32),
) -> Union[MAGNet, FlowMAGNet]:
    """
    Load model from collection / id and set inference parameters
    """
    torch.manual_seed(seed_model)

    # create dummy datamodule, this is only for things like id-to-shape maps and not to utilize the DL
    dm = MolDataModule(dataset, batch_size=128, num_workers=3, shuffle=False)
    dm.setup()

    # load model config from file
    checkpoint_dir = WB_LOG_DIR / collection / run_id / "checkpoints"
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

    # set dataset specific inference parameters
    print(
        "ATTENTION! Setting the following min/max inference sizes:",
        dm.feature_sizes["min_size"],
        dm.feature_sizes["max_size"],
    )
    setattr(model, "min_size", dm.feature_sizes["min_size"])
    setattr(model, "max_size", dm.feature_sizes["max_size"])

    # set inference parameters according to dataset default
    for key, value in sampling_params.items():
        setattr(model, key, value)
    if return_config:
        config.update(load_config)
        return model, config
    return model
