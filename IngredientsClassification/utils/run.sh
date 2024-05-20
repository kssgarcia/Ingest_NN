#!/bin/bash

#SBATCH --job-name=run_vit      # Job name (change to your preferred name)
#SBATCH --mail-type=FAIL,END          # Mail notification
#SBATCH --mail-user=kssepulveg@eafit.edu.co # User Email
#SBATCH --output=output.log           # Output log file
#SBATCH --error=error.log             # Error log file
#SBATCH --partition=longjobs          # SLURM partition/queue to submit the job
#SBATCH --nodes=1                     # Number of nodes required
#SBATCH --ntasks=1                    # Number of tasks (one task per node)
#SBATCH --cpus-per-task=12            # Number of CPU cores per task
#SBATCH --time=56:00:00               # Expected time for job completion (HH:MM:SS)

source ~/miniconda3/bin/activate testenv

cd /home/kssepulveg/project/SnapEat/ViT

python ViT_Food101.py
