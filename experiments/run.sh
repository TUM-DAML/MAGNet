#!/bin/bash

#SBATCH -N 1 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_all 
#SBATCH -t 1-00:00 
#SBATCH -o "slurm_psvae2.out"
#SBATCH --mem=32000 
#SBATCH --qos=phd 
#SBATCH --cpus-per-task=2
#SBATCH --exclude=gpu08
 
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate baselines
 
python guacamol_distribution.py --model_name PSVAE --model_id 1948p20m