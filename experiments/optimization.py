import argparse
import json
import time

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation

from models.global_utils import (
    BASELINE_DIR,
    SUPPORTED_DATA,
    SUPPORTED_MODELS,
    get_all_model_funcs,
)
from models.optimizer import GoalDirectedWrapper


def run_optimization(model_id, dataset, model_name, seed, opt_config, guacamol_benchmark, optimization_method):
    # load base model and inference server
    all_funcs = get_all_model_funcs(model_name)
    inference_server = all_funcs["load"](dataset=dataset, model_id=model_id, seed=seed)

    # prepare goal directed wrapper
    goal_directed_generator = GoalDirectedWrapper(
        inference_server, opt_config, dataset, guacamol_benchmark, optimization_method
    )

    # collect benchmark resultss
    json_path = BASELINE_DIR / "wb_logs" / "guacamol_jsons" / (str(time.time()) + ".json")
    assess_goal_directed_generation(
        goal_directed_generator, benchmark_version=guacamol_benchmark, json_output_file=json_path
    )
    inference_server.end_inference_server()
    results = dict()
    with open(json_path) as json_file:
        guacamol_results = json.load(json_file)
        for subdict in guacamol_results["results"]:
            results[subdict["benchmark_name"]] = subdict["score"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS, default="PSVAE")
    parser.add_argument("--model_id", type=str, default="bj01bmmh")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATA, default="zinc")
    parser.add_argument("--guacamol_benchmark", type=str, choices=["trivial", "v1", "v2"], default="trivial")
    parser.add_argument("--optimization_method", type=str, choices=["mso", "gasc"], default="mso")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opt_config",
        type=dict,
        default=dict(
            mso=dict(
                num_starting_mols=1,
                num_particles=50,
                num_swarms=1,
                num_runs=10,
                reencode_output=False,
            ),
            gasc=dict(num_steps=10, lr=0.1),
            num_train_samples=100,
        ),
    )
    args = parser.parse_args()

    result = run_optimization(
        args.model_id,
        args.dataset,
        args.model_name,
        args.seed,
        args.opt_config,
        args.guacamol_benchmark,
        args.optimization_method,
    )
    print(result)
