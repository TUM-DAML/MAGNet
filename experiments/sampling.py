import argparse

from baselines.global_utils import SUPPORTED_DATA, SUPPORTED_MODELS, get_all_model_funcs


def run_sampling(model_id, dataset, model_name, num_samples, seed):
    all_funcs = get_all_model_funcs(model_name)
    inference_server = all_funcs["load"](dataset=dataset, model_id=model_id, seed=seed)
    sampled_smiles = inference_server.sample_molecules(num_samples)
    return sampled_smiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATA, default="zinc")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_sampling(args.model_id, args.dataset, args.model_name, args.num_samples, args.seed)
    print(result)
