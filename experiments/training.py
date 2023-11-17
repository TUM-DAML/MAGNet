import argparse

from baselines.global_utils import SUPPORTED_DATA, SUPPORTED_MODELS, get_all_model_funcs


def training(seed, dataset, model_name):
    all_funcs = get_all_model_funcs(model_name)
    all_funcs["train"](seed, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATA, default="zinc")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    training(model_name=args.model_name, dataset=args.dataset, seed=args.seed)
