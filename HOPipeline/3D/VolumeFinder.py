import pyvista as pv
import numpy as np
import glob
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

#Mesh Folder Number (Starts at 0)
file_num = 0

#Change File Paths Accordingly
general_path = "/users/alanh/Documents/CBLResearch-github/HOPipeline/3D"
mesh_path = "MeshCases"
results_path = "results_HO"

def volume_extract(file_num):
    # Load reference meshes to find endo node indices
    ref_volume = pv.read(f'{general_path}/{mesh_path}/case_{file_num}/mesh_{file_num}_volume.vtu')
    endo_ref = pv.read(f'{general_path}/{mesh_path}/case_{file_num}/mesh_{file_num}_endo.vtp')



    ref_points = np.round(ref_volume.points, 4)
    endo_points = np.round(endo_ref.points, 4)
    endo_set = set(map(tuple, endo_points))
    endo_indices = np.array([i for i, p in enumerate(ref_points)
                            if tuple(p) in endo_set])
    print(f"Found {len(endo_indices)} endo nodes")


    #Rename files if they contain the wrong name
    # folder = f"{mesh_path}/results_{material_type}/"
    # for filename in os.listdir(folder):
    #     if filename.startswith(f"results_{material_type}_") and filename.endswith(".vtu"):
    #         new_filename = f"result_{material_type}" + filename[len(f"result_{material_type}_"):]
    #         os.rename(
    #             os.path.join(folder, filename),
    #             os.path.join(folder, new_filename)
    #     )
    #         print(f"Renamed: {filename} → {new_filename}")

    # --- MODIFIED HERE: Sliced [1:-1] to exclude first and last files ---
    files = sorted(glob.glob(f'{general_path}/{results_path}/Result_*.vtu'))

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

    np.savetxt(f'lv_volumes_{file_num}.csv', np.array(volumes), fmt='%.2f')
    return volumes

# Plotting generated volume vs. time data
# (times will automatically adjust to the new, shorter length of volumes)
volumesFill = volume_extract(file_num)
# volumesDivergence = volume_extract_divergence(file_num, "HO")
