#!/bin/bash
# usage: sbatch --job-name=<name> slurm/run.sbatch "<command>"
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nobackup/proj/disk/dionysus/shared/cuneiform-stoicheia/logs/%x-%j.log
source /nobackup/proj/disk/dionysus/shared/cuneiform-stoicheia/slurm/env.sh
echo "[job] $(hostname): $1"
eval "$1"
echo "[job] exit $?"
