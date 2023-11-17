import baselines.moler.molecule_generation.chem.motif_utils as motif_utils
from baselines.global_utils import BASELINE_DIR
from baselines.moler.molecule_generation.cli.preprocess import run_smiles_preprocessing
from baselines.moler.molecule_generation.cli.train import get_argparser, run_from_args
from baselines.moler.molecule_generation.utils.cli_utils import setup_logging

motif_args = dict(
    motif_keep_leaf_edges=False,
    motif_max_vocab_size=350,
    motif_min_frequency=None,
    motif_min_num_atoms=3,
)
motif_extraction_settings = motif_utils.MotifExtractionSettings(
    min_frequency=motif_args["motif_min_frequency"],
    min_num_atoms=motif_args["motif_min_num_atoms"],
    cut_leaf_edges=not motif_args["motif_keep_leaf_edges"],
    max_vocab_size=motif_args["motif_max_vocab_size"],
)

dataset = "zinc"
preproc_name = "moler_preproc_" + str(motif_extraction_settings.max_vocab_size)
input_dir = BASELINE_DIR / "smiles_files" / dataset
output_dir = BASELINE_DIR / "data" / "MOLER" / dataset / preproc_name
default_args = dict(
    input_dir=input_dir,
    output_dir=output_dir / "out",
    trace_dir=output_dir / "trace",
    generation_order="bfs-random",
    motif_extraction_settings=motif_extraction_settings,
    num_datapoints=None,
    num_processes=8,
    quiet=False,
    for_cgvae=False,
)


if __name__ == "__main__":
    run_smiles_preprocessing(**default_args)
