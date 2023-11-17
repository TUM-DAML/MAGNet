import argparse

from models.global_utils import (
    BASELINE_DIR,
    SUPPORTED_DATA,
    SUPPORTED_MODELS,
    get_all_model_funcs,
    smiles_from_file,
)


def run_reconstruction(model_id, dataset, model_name, num_samples, input_smiles_file):
    # prepare model and input smiles
    input_smiles_file = BASELINE_DIR / input_smiles_file
    all_funcs = get_all_model_funcs(model_name)
    inference_server = all_funcs["load"](dataset=dataset, model_id=model_id, seed=0)
    input_smiles = smiles_from_file(input_smiles_file)
    input_smiles = input_smiles[:num_samples]
    reconstructed_smiles = inference_server.reconstruct_molecules(input_smiles)
    return dict(reconstructed_smiles=reconstructed_smiles, gt_smiles=input_smiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS)
    parser.add_argument("--model_id", type=str)
    parser.add_argument(
        "--input_smiles_file",
        type=str,
        default="smiles_files/zinc/val.txt",
    )
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATA, default="zinc")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    result = run_reconstruction(
        args.model_id,
        args.dataset,
        args.model_name,
        args.num_samples,
        args.input_smiles_file,
    )
    print(result)
