import sys

from sacred import Experiment

import seml

from sampling import run_sampling

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
def run(num_samples, seed, dataset, model_config):
    model_name, model_id = model_config
    model_id = str(model_id)
    result = run_sampling(
        num_samples=num_samples,
        seed=seed,
        dataset=dataset,
        model_id=model_id,
        model_name=model_name,
    )
    return result
