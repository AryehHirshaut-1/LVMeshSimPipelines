import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re

#Change these to wanted pressure volume data
volumes_dir = r"C:\Users\alanh\Documents\CBLResearch-github\VolumeConvergence\results"
pressure_path = r"C:\Users\alanh\Documents\CBLResearch-github\dat\lv_pressure.dat"

# Find all case_*.csv files and sort them numerically by case number
csv_files = glob.glob(os.path.join(volumes_dir, "case_*.csv"))
csv_files.sort(key=lambda f: int(re.search(r"case_(\d+)\.csv", f).group(1)))

# Pressure file is the same for every case, so read it once outside the loop
pressurefile = pd.read_csv(pressure_path, sep=r'\s+', header=0)
pressures = pressurefile.iloc[:, 1]

plt.figure()

for csv_path in csv_files:
    case_num = int(re.search(r"case_(\d+)\.csv", csv_path).group(1))

    volumes_df = pd.read_csv(csv_path, header=None, names=["volume"])
    volumes = volumes_df["volume"]

    # Align lengths in case volumes/pressures differ by one row (as seen before)
    n = min(len(volumes), len(pressures))
    v = volumes.iloc[:n]
    p = pressures.iloc[:n]

    plt.plot(v, p, markersize=3, label=f"Case {case_num}")

plt.xlabel('Volume (cm³)')
plt.ylabel('Pressure (dyne/cm²)')
plt.title("Pressure Volume Loops — All Cases")
plt.grid(True)
plt.legend()

#Change this line to choose where file is saved
plt.savefig(r"C:\Users\alanh\Documents\CBLResearch-github\VolumeConvergence\PVLoop_VolCon.png", dpi=300)
plt.show()