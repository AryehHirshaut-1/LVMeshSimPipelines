import glob
import os
import re
import warnings
import numpy as np
import pyvista as pv

#For visualization
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

file_num = 0
def volume_extract(file_num):
    # Load reference meshes to find endo node indices
    ref_volume = pv.read(f'/users/alanh/Documents/CBLResearch-github/3DModeling/ActiveContraction/lv_sim_cases/ActiveSimCase/case_{file_num}/mesh_{file_num}_volume.vtu')
    endo_ref = pv.read(f'/users/alanh/Documents/CBLResearch-github/3DModeling/ActiveContraction/lv_sim_cases/ActiveSimCase/case_{file_num}/mesh_{file_num}_endo.vtp')



    ref_points = np.round(ref_volume.points, 4)
    endo_points = np.round(endo_ref.points, 4)
    endo_set = set(map(tuple, endo_points))
    endo_indices = np.array([i for i, p in enumerate(ref_points)
                            if tuple(p) in endo_set])
    print(f"Found {len(endo_indices)} endo nodes")


    #Rename files if they contain the wrong name
    folder = f"/users/alanh/Documents/CBLResearch-github/3DModeling/ActiveContraction/ACresults/ActiveSimNVResults/"
    for filename in os.listdir(folder):
        if filename.startswith(f"results_") and filename.endswith(".vtu"):
            new_filename = f"result" + filename[len(f"result"):]
            os.rename(
                os.path.join(folder, filename),
                os.path.join(folder, new_filename)
        )
            print(f"Renamed: {filename} → {new_filename}")

    # --- MODIFIED HERE: Sliced [1:-1] to exclude first and last files ---
    files = sorted(glob.glob(f"/users/alanh/Documents/CBLResearch-github/3DModeling/ActiveContraction/ACresults/ActiveSimNVResults/ACresult_*.vtu"))[1:-1]

    volumes = []

    for f in files:
        mesh = pv.read(f)

        # Warp the mesh by displacement to get true deformed geometry
        displacement = mesh.point_data['Displacement']
        deformed = mesh.copy(deep=True)
        deformed.points = mesh.points + displacement

        # Extract deformed surface
        surface = deformed.extract_surface(algorithm='dataset_surface')
        orig_ids = surface.point_data['vtkOriginalPointIds']

        endo_index_set = set(endo_indices)
        surface_endo_mask = np.array([pid in endo_index_set for pid in orig_ids])

        endo_surf = surface.extract_points(surface_endo_mask, adjacent_cells=True)
        endo_surf = endo_surf.extract_surface()
        endo_surf = endo_surf.triangulate()

        endo_closed = endo_surf.fill_holes(1000)
        endo_closed = endo_closed.triangulate()

        vol = endo_closed.volume
        volumes.append(vol)

    np.savetxt(f"/users/alanh/Documents/CBLResearch-github/3DModeling/ActiveContraction/dat/ACNVLV_extractedvolumes_{file_num}.csv", np.array(volumes), fmt='%.2f')
    return volumes


volumes_dir = r"C:\Users\alanh\Documents\CBLResearch-github\3DModeling/ActiveContraction"
pressure_path = r"C:\Users\alanh\Documents\CBLResearch-github\3DModeling\ActiveContraction\dat\pressure_scaled.dat"

pressurefile = pd.read_csv(pressure_path, sep=r'\s+', header=0)
pressures = pressurefile.iloc[:, 1]

volumes = volume_extract(0)
times = np.arange(1, len(volumes) + 1) * 0.00813



# Align lengths in case volumes/pressures differ by one row (as seen before)
# n = min(len(volumes), len(pressures))
# v = volumes[:n]
# p = pressures[:n]

plt.plot(times, volumes, markersize=3)
plt.xlabel('Time (s)')
plt.ylabel('Volume (cm³)')
plt.title("Volume vs Time")
plt.grid(True)

#Change this line to choose where file is saved
plt.savefig(r"C:\Users\alanh\Documents\CBLResearch-github\3DModeling\ActiveContraction\ACNVPVLoop.png", dpi=300)
#plt.show()