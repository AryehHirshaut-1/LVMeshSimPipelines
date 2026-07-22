#!/bin/bash
#SBATCH --job-name=lv_sensitivity_analysis
#SBATCH --array=0-809
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/case_%a.out
#SBATCH --error=logs/case_%a.err

module load svMultiPhysics/March_2025-foss-2024a
module load Python/3.12.3-GCCcore-13.3.0

set -euo pipefail

# Activate your virtual environment
source ~/NHPipeline/3D/14/bin/activate

# ===== EDIT FOR YOUR CLUSTER =====================================
WORKDIR="$HOME/NHPipeline/3D"
TEMPLATE="$HOME/NHPipeline/3D/lv_sim_nh_HPC.xml"

i="$SLURM_ARRAY_TASK_ID"
modulus=$(python3 -c "
import pyvista as pv
mesh = pv.read('MeshCasesPig/case_${i}/mesh_${i}_volume.vtu')
print(mesh.cell_data['ElasticityModulus'][0])
")

cd "$WORKDIR"
mkdir -p inputs logs "results/case_${i}"

XML="inputs/lv_sim_nh_case_${i}.xml"
sed "s/CASEID/${i}/g; s/ELAS_MOD/${modulus}/g" "$TEMPLATE" > "$XML"

echo "[case ${i}] start $(date) host=$(hostname) ranks=${SLURM_NTASKS} xml=${XML}"

mpirun -np 3 svmultiphysics "$XML"

echo "[case ${i}] done  $(date) -> results/case_${i}/"

source 14/bin/activate

# Volume extraction for this case only
python3 "$WORKDIR/VolumeFinder.py" \
    --case "$i" \
    --general-path "$WORKDIR" \
    --mesh-path "MeshCasesPig" \
    --results-path "MultiPhysicsResults"

# ---- Free disk: drop this case's raw svMultiPhysics results after volumes are extracted ----
# We only need MeshCases; extracted volumes are saved in VolumeResults/lv_volumes_<i>.csv
VOLCSV="$WORKDIR/VolumeResults/lv_volumes_${i}.csv"
if [ -s "$VOLCSV" ]; then
    echo "[case ${i}] volumes extracted OK -> removing raw results MultiPhysicsResults/case_${i}"
    rm -rf "$WORKDIR/MultiPhysicsResults/case_${i}"
else
    echo "[case ${i}] ERROR: ${VOLCSV} missing/empty -> keeping raw results for debugging" >&2
    exit 1
fi

