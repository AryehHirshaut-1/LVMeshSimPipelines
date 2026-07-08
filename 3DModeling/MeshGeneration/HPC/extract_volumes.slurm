#!/bin/bash
#SBATCH --job-name=vol_extract
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/vol_%a.out
#SBATCH --error=logs/vol_%a.err

source /etc/profile
module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/volenv/bin/activate"

set -euo pipefail
cd "$HOME/CBLResearch-github"
mkdir -p logs

python extract_volumes.py "$SLURM_ARRAY_TASK_ID" HO "$HOME/CBLResearch-github/extracted volume data"
