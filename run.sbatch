#!/bin/bash

# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --job-name="Hudai"              # change job name
#SBATCH --time=0-01:00:00               # walltime limit (day-HH:MM:SS)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --cpus-per-task=4               # number of processes
#SBATCH --mem=1200G
#SBATCH --account=cdsteven-lab
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/job.%J.out        # tell it to store the output console text to a file called job.<assigned job number>.out
#SBATCH --error=logs/job.%J.err         # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err

# SBATCH --mail-user=jubair@iastate.edu
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --ntasks-per-node=4   # 4 processor core(s) per node
#SBATCH --exclude=nova21-gpu-10    #nova22-amp-8,nova21-gpu-3,nova21-gpu-1,nova21-gpu-11


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source /lustre/hdd/LAS/cdsteven-lab/jubair/research/LLMScratch/.venv/bin/activate
export HF_HOME=/lustre/hdd/LAS/cdsteven-lab/jubair/research/.cache/huggingface

root="/lustre/hdd/LAS/cdsteven-lab/jubair/research/LLMScratch"

echo "Job started on $SLURM_NODELIST and time:  $(date +%T)"
nvidia-smi 

torchrun --nproc_per_node=2 DDP_torchrun.py

echo "Job done"