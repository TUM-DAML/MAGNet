import seml
from sacred import Experiment
from training import training

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
def run(model_seed, dataset, model_name):
    training(model_name=model_name, seed=model_seed, dataset=dataset)
    return dict
