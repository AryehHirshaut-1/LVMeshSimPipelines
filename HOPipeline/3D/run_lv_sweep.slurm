#!/bin/bash
#SBATCH --job-name=lv_sensitivity_analysis
#SBATCH --array=0-99           # one task per case: case_0 ... case_99  (use 0-99%10 to cap concurrency)
#SBATCH --nodes=1
#SBATCH --ntasks=16             # MPI ranks per case  -> tune to your mesh size / node
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00        # wall-clock PER case
#SBATCH --output=logs/case_%a.out
#SBATCH --error=logs/case_%a.err

module load svMultiPhysics/March_2025-foss-2024a
module load /apps/software/2024a/software/Python/3.12.3-GCCcore-13.3.0/bin/python3

set -euo pipefail


# ===== EDIT FOR YOUR CLUSTER =====================================
WORKDIR="$HOME/HOPipeline/3D"                                  # holds lv_sim_cases/, lv_pressure.dat, the template
TEMPLATE="$HOME/HOPipeline/3D/lv_sim_ho_HPC.xml"          # uncomment / adjust to load the MPI used to build svMultiPhysics

i="$SLURM_ARRAY_TASK_ID"
cd "$WORKDIR"
mkdir -p inputs logs "results/case_${i}"

# Build this case's input by substituting the case number into the template.
XML="inputs/lv_sim_ho_case_${i}.xml"
sed "s/CASEID/${i}/g" "$TEMPLATE" > "$XML"

echo "[case ${i}] start $(date) host=$(hostname) ranks=${SLURM_NTASKS} xml=${XML}"

# srun is the SLURM-native MPI launcher. If your site needs mpirun instead, use:
#   mpirun -n "$SLURM_NTASKS" "$SOLVER" "$XML"
mpirun -np 3 svmultiphysics "$XML"

echo "[case ${i}] done  $(date) -> results/case_${i}/"

# Add VolumeFinder runner after
