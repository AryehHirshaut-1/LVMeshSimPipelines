#!/bin/bash
# Optional: pre-generate one concrete XML per case (case_0 .. case_99) into inputs/.
# You do NOT need this if you use run_lv_sweep.slurm, which substitutes on the fly.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p inputs
N="${1:-100}"                      # number of cases (default 100 -> indices 0..N-1)
for i in $(seq 0 $((N-1))); do
  sed "s/CASEID/${i}/g" lv_sim_nh_HPC.xml > "inputs/lv_sim_nh_case_${i}.xml"
done
echo "Wrote ${N} files to inputs/  (inputs/lv_sim_nh_case_0.xml .. inputs/lv_sim_nh_case_$((N-1)).xml)"
