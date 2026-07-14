#!/bin/bash
#SBATCH --job-name=lv_sensitivity_analysis
#SBATCH --array=0-89
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/case_%a.out
#SBATCH --error=logs/case_%a.err

module load svMultiPhysics/March_2025-foss-2024a
module load Python/3.12.3-GCCcore-13.3.0

set -euo pipefail

# Activate your virtual environment
source ~/HOPipeline/3D/14/bin/activate

# ===== EDIT FOR YOUR CLUSTER =====================================
WORKDIR="$HOME/HOPipeline/3D"
TEMPLATE="$HOME/HOPipeline/3D/lv_sim_ho_HPC.xml"

i="$SLURM_ARRAY_TASK_ID"
cd "$WORKDIR"
mkdir -p inputs logs "results/case_${i}"

XML="inputs/lv_sim_ho_case_${i}.xml"
sed "s/CASEID/${i}/g" "$TEMPLATE" > "$XML"

echo "[case ${i}] start $(date) host=$(hostname) ranks=${SLURM_NTASKS} xml=${XML}"

mpirun -np 3 svmultiphysics "$XML"

echo "[case ${i}] done  $(date) -> results/case_${i}/"

# Volume extraction for this case only
python3 "$WORKDIR/VolumeFinder.py" \
    --case "$i" \
    --general-path "$WORKDIR" \
    --mesh-path "MeshCases" \
    --results-path "MultiPhysicsResults"
