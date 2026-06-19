import pyvista as pv
import numpy as np
import glob
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

#Mesh Folder Number (Starts at 0)
#Material Type - Either NH (Neo-Hookean) or HO (Holzapfel-Ogden)
file_num = 0
material_type = "HO"

def volume_extract(file_num, material_type):
    # Load reference meshes to find endo node indices
    ref_volume = pv.read(f'/users/alanh/Documents/CBLResearch-github/lv_sim_cases/case_{file_num}/mesh_{file_num}_volume.vtu')
    endo_ref = pv.read(f'/users/alanh/Documents/CBLResearch-github/lv_sim_cases/case_{file_num}/mesh_{file_num}_endo.vtp')



    ref_points = np.round(ref_volume.points, 4)
    endo_points = np.round(endo_ref.points, 4)
    endo_set = set(map(tuple, endo_points))
    endo_indices = np.array([i for i, p in enumerate(ref_points)
                            if tuple(p) in endo_set])
    print(f"Found {len(endo_indices)} endo nodes")


    #Rename files if they contain the wrong name
    folder = f"/users/alanh/Documents/CBLResearch-github/results_{material_type}/"
    for filename in os.listdir(folder):
        if filename.startswith(f"results_{material_type}_") and filename.endswith(".vtu"):
            new_filename = f"result_{material_type}" + filename[len(f"result_{material_type}_"):]
            os.rename(
                os.path.join(folder, filename),
                os.path.join(folder, new_filename)
        )
            print(f"Renamed: {filename} → {new_filename}")

    # --- MODIFIED HERE: Sliced [1:-1] to exclude first and last files ---
    files = sorted(glob.glob(f'/users/alanh/Documents/CBLResearch-github/results_{material_type}/result_{material_type}_*.vtu'))[1:-1]

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

    np.savetxt(f'/users/alanh/Documents/CBLResearch-github/lv_volumes_{file_num}_{material_type}.csv', np.array(volumes), fmt='%.2f')
    return volumes


# Plotting generated volume vs. time data
# (times will automatically adjust to the new, shorter length of volumes)
volumes = volume_extract(file_num, "HO")

times = np.arange(1, len(volumes) + 1) * 0.00813

dat_times =     [0.00000,  0.87800, 1.61780]
dat_pressures = [0.0,     67273.61, 0.0]
pressures = np.interp(times, dat_times, dat_pressures)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(volumes, pressures, markersize=3, label="NH")
#ax1.plot(volumes1, pressures, markersize=3, color="orange", label="HO")
ax1.set_xlabel('Volume (cm³)')
ax1.set_ylabel('Pressure (dyne/cm²)')
ax1.set_title('Pressure-Volume Loop')
ax1.grid(True)
ax1.legend()

ax2.plot(times, volumes, markersize=3)
#ax2.plot(times, volumes1, color='orange', markersize=3)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Volume (cm³)')
ax2.set_title('Volume vs. Time')
ax2.grid(True)

plt.savefig('/users/alanh/Documents/CBLResearch-github/pv_loop.png', dpi=300)
plt.show()