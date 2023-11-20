import sys

from sacred import Experiment

import seml

sys.path.append(".")
from optimization import run_optimization

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(model_config, dataset, seed, opt_config, guacamol_benchmark, optimization_method):
    model_name, model_id = model_config
    model_id = str(model_id)
    result = run_optimization(
        model_id=model_id,
        dataset=dataset,
        model_name=model_name,
        seed=seed,
        opt_config=opt_config,
        guacamol_benchmark=guacamol_benchmark,
        optimization_method=optimization_method,
    )
    return result