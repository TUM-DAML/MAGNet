import argparse

from models.global_utils import SUPPORTED_DATA, SUPPORTED_MODELS, get_all_model_funcs
from models.datasets import check_dataset_exists


def run_preprocessing(model_name, dataset, num_workers):
    check_dataset_exists(dataset)
    get_all_model_funcs(model_name)["preprocessing"](dataset, num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS, default="MAGNET")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATA, default="zinc")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    result = run_preprocessing(args.model_name, args.dataset, args.num_workers)
    print(result)
