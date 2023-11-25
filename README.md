# MAGNet: Motif-Agnostic Generation of Molecules from Shapes
Recent advances in machine learning for molecules exhibit great potential for facilitating drug discovery from in silico predictions. Most models for molecule generation rely on the decomposition of molecules into frequently occurring substructures (motifs), from which they generate novel compounds. While motif representations greatly aid in learning molecular distributions, such methods struggle to represent substructures beyond their known motif set. To alleviate this issue and increase flexibility across datasets, we propose MAGNet, a graph-based model that generates abstract shapes before allocating atom and bond types. To this end, we introduce a novel factorisation of the molecules' data distribution that accounts for the molecules' global context and facilitates learning adequate assignments of atoms and bonds onto shapes. Despite the added complexity of shape abstractions, MAGNet outperforms most other graph-based approaches on standard benchmarks. Importantly, we demonstrate that MAGNet's improved expressivity leads to molecules with more topologically distinct structures and, at the same time, diverse atom and bond assignments. 

![](MAGNet.png)

If you use this repository for your research, please cite:
```
@article{hetzel2023magnet,
  title={MAGNet: Motif-Agnostic Generation of Molecules from Shapes},
  author={Hetzel, Leon and Sommer, Johanna and Rieck, Bastian and Theis, Fabian and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2305.19303},
  year={2023}
}
```


# Molecule Generation Baselines
Besides the MAGNet code, we provide the same interface, package structure and conda environment for all evaluated baselines in hopes of facilitating ease of reproducibility.

Supported Molecule Generation Models:
- [✨ MAGNET ✨](https://arxiv.org/abs/2305.19303)
- [SMILES-LSTM](https://arxiv.org/abs/1701.01329)
- [CHARVAE](https://arxiv.org/pdf/1610.02415.pdf) (without Terminal GRU)
- [JTVAE](https://arxiv.org/abs/1802.04364)
- [HIERVAE](https://arxiv.org/pdf/2002.03230.pdf)
- [PSVAE](https://arxiv.org/abs/2106.15098)
- [MOLER](https://arxiv.org/abs/2103.03864)
- [GRAPHAF](https://proceedings.neurips.cc/paper_files/paper/2018/file/d60678e8f2ba9c540798ebbde31177e8-Paper.pdf)
- [GCPN](https://arxiv.org/pdf/2001.09382.pdf)

This repository does not reimplement the models but only provides a wrapper / unified interface to access them. We would thus like to thank the authors of the aforementioned papers for making their code public.

## Setup

As a first step, please create the conda environment with
```
mamba env create -f config/env.yaml
conda activate baselines
pip install -e .
```

Currently, there are severe problems with installing `torchdyn`. For the time being, you can do this via
```
pip install -c config/constraints.txt git+https://github.com/DiffEqML/torchdyn.git
```

Either you have cloned the repository with the command `git clone xyz --recurse-submodules` and have the code for required resources, or you can get neccessary repositories by interating through the `resources/` folder with the commands 
```
git submodule init
git submodule update
pip install -e .
```

Further, please adjust the path configuration at  `baselines/global_utils.py` according to your file structure as well as W&B config.

## General Structure

We have currently structured the repository s.t. wrappers exist for the individual baselines and **training, sampling, reconstruction and optimization** can be done via the following files:

```
# Example: sampling, this file structure exists for all task
experiments/sampling.py # script for direct execution
# If you are using SEML
experiments/sampling_seml.yaml # yaml config to execute sampling
experiments/sampling_seml.py # seml execution file
```

Additional Evaluations:
- `evaluations/guacamol_benchmark.py` to evaluate MOSES and GUACAMOL distribution learning benchmarks (given that you have run a sampling SEML file beforehand, which you are asked to specify)
- `evaluations/zero_shot_generalization.py` to evaluate reconstruction performance (given that you have run a reconstruction SEML file beforehand, which you are asked to specify)
- `evaluations/interpolation.ipynb` to plot interpolation in the latent space

What is currently not supported is a central means of starting preprocessing pipelines for the individual models. Please refer to i.e. `baselines/JTVAE/preprocess_jtvae.py` to start the data preprocessing for JTVAE.
