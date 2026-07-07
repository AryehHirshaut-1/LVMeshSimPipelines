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
general_path = "/users/alanh/Documents/CBLResearch-github/ActiveContraction"
mesh_path = "lv_sim_cases/ActiveSimCase"
results_path = "ACResults/ActiveSimNVResults/ACresult"

def volume_extract(file_num, material_type):
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
    files = sorted(glob.glob(f'{general_path}/{results_path}_*.vtu'))[1:-1]

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

    np.savetxt(f'{mesh_path}/lv_volumes_{file_num}.csv', np.array(volumes), fmt='%.2f')
    return volumes

def volume_divergence_theorem(surface: pv.PolyData) -> float:
    """
    Compute the enclosed volume of a CLOSED, triangulated surface using the
    divergence theorem (signed tetrahedra formula):
 
        V = (1/6) * sum_over_triangles( v0 . (v1 x v2) )
 
    where v0, v1, v2 are triangle vertices as position vectors from the
    origin. Exact for any watertight, consistently-oriented (outward normals)
    triangulated surface -- no grid, no sampling resolution to tune.
 
    Parameters
    ----------
    surface : pv.PolyData
        Must already be closed/watertight and triangulated (e.g. the output
        of .fill_holes().triangulate()).
 
    Returns
    -------
    float
        Signed volume. Take abs() if the surface turns out to be inward-
        oriented (negative result).
    """
    pts = surface.points
    faces = surface.faces.reshape(-1, 4)[:, 1:4]  # drop leading "3" per triangle
 
    v0 = pts[faces[:, 0]]
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]
 
    cross = np.cross(v1, v2)
    signed_vols = np.einsum('ij,ij->i', v0, cross)
 
    return signed_vols.sum() / 6.0

def volume_extract_divergence(file_num, material_type):
    # Load reference meshes to find endo node indices
    ref_volume = pv.read(f'/users/alanh/Documents/CBLResearch-github/MeshGeneration/lv_sim_cases/case_{file_num}/mesh_{file_num}_volume.vtu')
    endo_ref = pv.read(f'/users/alanh/Documents/CBLResearch-github/MeshGeneration/lv_sim_cases/case_{file_num}/mesh_{file_num}_endo.vtp')
 
    ref_points = np.round(ref_volume.points, 4)
    endo_points = np.round(endo_ref.points, 4)
    endo_set = set(map(tuple, endo_points))
    endo_indices = np.array([i for i, p in enumerate(ref_points)
                            if tuple(p) in endo_set])
    print(f"Found {len(endo_indices)} endo nodes")
 
    #Rename files if they contain the wrong name
    folder = f"/users/alanh/Documents/CBLResearch-github/MeshGeneration/results_{material_type}/"
    for filename in os.listdir(folder):
        if filename.startswith(f"results_{material_type}_") and filename.endswith(".vtu"):
            new_filename = f"result_{material_type}" + filename[len(f"result_{material_type}_"):]
            os.rename(
                os.path.join(folder, filename),
                os.path.join(folder, new_filename)
        )
            print(f"Renamed: {filename} → {new_filename}")
 
    # --- Sliced [1:-1] to exclude first and last files ---
    files = sorted(glob.glob(f'/users/alanh/Documents/CBLResearch-github/MeshGeneration/results_{material_type}/result_{material_type}_*.vtu'))[1:-1]
    print(f"Found {len(files)} timestep files")
 
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
 
        vol = abs(volume_divergence_theorem(endo_closed))
        volumes.append(vol)
 
    np.savetxt(f'/users/alanh/Documents/CBLResearch-github/VolumeExtractions/lv_volumes_{file_num}_{material_type}_divergence.csv', np.array(volumes), fmt='%.2f')
    return volumes
 


# Plotting generated volume vs. time data
# (times will automatically adjust to the new, shorter length of volumes)
volumesFill = volume_extract(file_num, "HO")
# volumesDivergence = volume_extract_divergence(file_num, "HO")


timesho = np.arange(1, len(volumesFill) + 1) * 0.00813
dat_times =     [0.00000,  1.61780]

plt.plot(timesho, volumesFill, markersize=3, label="Fill Holes")
# plt.plot(timesho, volumesDivergence, markersize=3, label='Divergence Theorem')
#ax1.plot(volumes1, pressures, markersize=3, color="orange", label="HO")
plt.xlabel('Time (s)')
plt.ylabel('Volume (cm³)')
plt.title('Volume vs. Time')
plt.grid(True)



plt.savefig(f'{mesh_path}/Extraction.png', dpi=300)
plt.show()