#!/bin/bash
# Example Slurm job for UMich Great Lakes: full study with the optimised parallel code.
# Submit from the repository root:   sbatch scripts/greatlakes_parallel.sbatch
#SBATCH --job-name=gvae-sim
#SBATCH --account=stats607f26_class      # <- replace with your Great Lakes account
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G                         # ~1 GB peak per worker at n = 1500
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

module load python/3.11
source .venv/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/time -v make parallel JOBS=$SLURM_CPUS_PER_TASK
