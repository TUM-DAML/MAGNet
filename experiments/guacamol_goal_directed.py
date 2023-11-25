import argparse
import json
import time

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation

from models.global_utils import (
    SUPPORTED_DATA,
    SUPPORTED_MODELS,
    get_all_model_funcs,
    WB_LOG_DIR
)
from models.wrappers.guacamol_benchmarks import GoalDirectedWrapper, read_results_from_file


def run_optimization(model_id, dataset, model_name, seed, opt_config, guacamol_benchmark, optimization_method):
    # load base model and inference server
    all_funcs = get_all_model_funcs(model_name)
    inference_server = all_funcs["load"](dataset=dataset, model_id=model_id, seed=seed)

    # prepare goal directed wrapper
    goal_directed_generator = GoalDirectedWrapper(
        inference_server, opt_config, dataset, guacamol_benchmark, optimization_method
    )

    # collect benchmark results
    json_path = WB_LOG_DIR / "guacamol_jsons" / (str(time.time()) + ".json")
    assess_goal_directed_generation(
        goal_directed_generator, benchmark_version=guacamol_benchmark, json_output_file=json_path
    )
    inference_server.end_inference_server()

    return read_results_from_file(str(json_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS)
    parser.add_argument("--model_id", type=str)
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
