#!/bin/bash
#SBATCH --job-name=lv_sensitivity_analysis
#SBATCH --array=0-99           # one task per case: case_0 ... case_99  (use 0-99%10 to cap concurrency)
#SBATCH --nodes=1
#SBATCH --ntasks=8             # MPI ranks per case  -> tune to your mesh size / node
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00        # wall-clock PER case
#SBATCH --mail-type=END
#SBATCH --mail-user=aryeh.hirshaut@yale.edu
#SBATCH --output=logs/case_%a.out
#SBATCH --error=logs/case_%a.err

module load svMultiPhysics/March_2025-foss-2024a

set -euo pipefail

# ===== EDIT FOR YOUR CLUSTER =====================================
WORKDIR="$HOME/CBLResearch"                                  # holds lv_sim_cases/, lv_pressure.dat, the template
SOLVER="$HOME/svMultiPhysics/svMultiPhysics-build/bin/svmultiphysics"
TEMPLATE="lv_sim_nh_HPC.xml"
module load openmpi          # uncomment / adjust to load the MPI used to build svMultiPhysics

i="$SLURM_ARRAY_TASK_ID"
cd "$WORKDIR"
mkdir -p inputs logs "results/case_${i}"

# Build this case's input by substituting the case number into the template.
XML="inputs/lv_sim_nh_case_${i}.xml"
sed "s/CASEID/${i}/g" "$TEMPLATE" > "$XML"

echo "[case ${i}] start $(date) host=$(hostname) ranks=${SLURM_NTASKS} xml=${XML}"

# srun is the SLURM-native MPI launcher. If your site needs mpirun instead, use:
#   mpirun -n "$SLURM_NTASKS" "$SOLVER" "$XML"
srun "$SOLVER" "$XML"

echo "[case ${i}] done  $(date) -> results/case_${i}/"
