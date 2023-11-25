import argparse
import time
import json
from models.global_utils import SUPPORTED_DATA, SUPPORTED_MODELS, get_all_model_funcs, WB_LOG_DIR, SMILES_DIR
from guacamol.assess_distribution_learning import assess_distribution_learning
from models.wrappers.guacamol_benchmarks import RandomGenerator, read_results_from_file

def run_sampling(model_id, dataset, model_name, gt_distribution, seed):
    # load base model and inference server
    all_funcs = get_all_model_funcs(model_name)
    inference_server = all_funcs["load"](dataset=dataset, model_id=model_id, seed=seed)
    
    # prepare generation wrapper
    random_generator = RandomGenerator(inference_server)
    
    # collect benchmark results
    gt_dist_path = SMILES_DIR / gt_distribution / "train.txt"
    json_path = WB_LOG_DIR / "guacamol_jsons" / (str(time.time()) + ".json")
    assess_distribution_learning(
        random_generator,
        chembl_training_file=str(gt_dist_path),
        json_output_file=json_path,
        benchmark_version="v2",
    )
    inference_server.end_inference_server()

    return read_results_from_file(str(json_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--training_dataset", type=str, choices=SUPPORTED_DATA, default="zinc")
    parser.add_argument("--gt_distribution", type=str, default="zinc")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_sampling(args.model_id, args.training_dataset, args.model_name, args.gt_distribution, args.seed)
    print(result)
