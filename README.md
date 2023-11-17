# Baselines for Molecule Generation
This repository is supposed to be a quick way for us & our students to get results for baselines. Implementing and reproducing baselines can take up too much time and our intention is to avoid this with this repository. The goal of the implementation is to keep the baseline code as close as possible to the code provided by the authors to ensure that no substantial errors are made in the reimplementation.

Currently, the following papers are supported:
- [✨ MAGNET ✨](https://arxiv.org/abs/2305.19303)
- [SMILES-LSTM](https://arxiv.org/abs/1701.01329)
- [CHARVAE](https://arxiv.org/pdf/1610.02415.pdf) (without Terminal GRU)
- [JTVAE](https://arxiv.org/abs/1802.04364)
- [HIERVAE](https://arxiv.org/pdf/2002.03230.pdf)
- [MICAM](https://arxiv.org/pdf/2302.01129.pdf) (Currently broken)
- [PSVAE](https://arxiv.org/abs/2106.15098)
- [MOLER](https://arxiv.org/abs/2103.03864)
- [GRAPHAF](https://proceedings.neurips.cc/paper_files/paper/2018/file/d60678e8f2ba9c540798ebbde31177e8-Paper.pdf)
- [GCPN](https://arxiv.org/pdf/2001.09382.pdf)

Inference (sampling & reconstruction) as well as training are supported in this repository. 

Please, if you encouter trouble using this repository, let us know. Based on your experience we can make the repository more accessible and prevent future students from facing the same issues as you. 

## General Structure

If you are a new student, please remind us to copy the project directory into your student directory on the server.

We have currently structure the repository s.t. wrappers exist for the individual baselines and training, sampling & reconstruction can be done via the following files:

```
# Example: sampling, files exist for all task
sampling.py # script for direct execution
seml/sampling_seml.yaml # yaml config to execute sampling
seml/sampling_seml.py # seml execution file
```

Further, you can add benchmark results to a given database with the `evaluations/guacamol_benchmark.py` script (given that you have run a sampling SEML file beforehand, which you are asked to specify) and evaluate reconstruction with `evaluations/zero_shot_generalization.py`.

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
