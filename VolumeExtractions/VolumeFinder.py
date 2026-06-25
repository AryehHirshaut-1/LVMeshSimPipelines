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

    np.savetxt(f'/users/alanh/Documents/CBLResearch-github/Volume_Extractions/lv_volumes_{file_num}_{material_type}.csv', np.array(volumes), fmt='%.2f')
    return volumes


# Plotting generated volume vs. time data
# (times will automatically adjust to the new, shorter length of volumes)
volumes = volume_extract(file_num, "NH")
volumes1 = volume_extract(file_num, "HO")

timesnh = np.arange(1, len(volumes1) + 1) * 0.00813
dat_times =     [0.00000,  1.61780]
dat_pressures = [0.0,     50000]
pressuresnh = np.interp(timesnh, dat_times, dat_pressures)

timesho = np.arange(1, len(volumes1) + 1) * 0.00813
dat_times =     [0.00000,  0.87800, 1.61780]
dat_pressures = [0.0,     67273.61, 0.0]
pressuresho = np.interp(timesho, dat_times, dat_pressures)


fig, axs = plt.subplots(2, 2, figsize=(10, 4))

axs[0,0].plot(volumes, pressuresnh, markersize=3, label="NH")
#ax1.plot(volumes1, pressures, markersize=3, color="orange", label="HO")
axs[0,0].set_xlabel('Volume (cm³)')
axs[0,0].set_ylabel('Pressure (dyne/cm²)')
axs[0,0].set_title('Pressure-Volume Loop')
axs[0,0].grid(True)

axs[0,1].plot(timesnh, volumes, markersize=3)
#ax2.plot(times, volumes1, color='orange', markersize=3)
axs[0,1].set_xlabel('Time (s)')
axs[0,1].set_ylabel('Volume (cm³)')
axs[0,1].set_title('Volume vs. Time')
axs[0,1].grid(True)

axs[1,0].plot(volumes1, pressuresho, markersize=3, color='orange')
axs[1,0].set_xlabel('Volume (cm³)')
axs[1,0].set_ylabel('Pressure (dyne/cm²)')
axs[1,0].grid(True)

axs[1,1].plot(timesho, volumes1, markersize=3, color='orange')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_ylabel('Volume (cm³)')
axs[1,1].grid(True)

plt.savefig('/users/alanh/Documents/CBLResearch-github/pv_loop.png', dpi=300)
plt.show()