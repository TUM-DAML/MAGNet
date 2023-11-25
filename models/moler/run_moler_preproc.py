import models.moler.molecule_generation.chem.motif_utils as motif_utils
from models.global_utils import BASELINE_DIR, get_model_config, SMILES_DIR, DATA_DIR
from models.moler.molecule_generation.cli.preprocess import run_smiles_preprocessing
from models.moler.molecule_generation.cli.train import get_argparser, run_from_args
from models.moler.molecule_generation.utils.cli_utils import setup_logging

def run_moler_preprocessing(dataset_name, num_processes):
    vocab_size = get_model_config("moler", dataset_name)["vocab_size"]
    motif_args = dict(
        motif_keep_leaf_edges=False,
        motif_max_vocab_size=vocab_size,
        motif_min_frequency=None,
        motif_min_num_atoms=3,
    )
    motif_extraction_settings = motif_utils.MotifExtractionSettings(
        min_frequency=motif_args["motif_min_frequency"],
        min_num_atoms=motif_args["motif_min_num_atoms"],
        cut_leaf_edges=not motif_args["motif_keep_leaf_edges"],
        max_vocab_size=motif_args["motif_max_vocab_size"],
    )

    preproc_name = "moler_preproc_" + str(motif_extraction_settings.max_vocab_size)
    input_dir = SMILES_DIR / dataset_name
    output_dir = DATA_DIR / "MOLER" / dataset_name / preproc_name
    default_args = dict(
        input_dir=input_dir,
        output_dir=output_dir / "out",
        trace_dir=output_dir / "trace",
        generation_order="bfs-random",
        motif_extraction_settings=motif_extraction_settings,
        num_datapoints=None,
        num_processes=num_processes,
        quiet=False,
        for_cgvae=False,
    )

    run_smiles_preprocessing(**default_args)
