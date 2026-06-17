import pyvista as pv
import numpy as np
import glob
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

# Load reference meshes to find endo node indices
ref_volume = pv.read('/users/alanh/Documents/CBLResearch/lv_sim_cases/case_1/mesh_1_volume.vtu')
endo_ref = pv.read('/users/alanh/Documents/CBLResearch/lv_sim_cases/case_1/mesh_1_endo.vtp')

ref_points = np.round(ref_volume.points, 4)
endo_points = np.round(endo_ref.points, 4)
endo_set = set(map(tuple, endo_points))
endo_indices = np.array([i for i, p in enumerate(ref_points)
                         if tuple(p) in endo_set])
print(f"Found {len(endo_indices)} endo nodes")

# --- MODIFIED HERE: Sliced [1:-1] to exclude first and last files ---
files = sorted(glob.glob('/users/alanh/Documents/CBLResearch/results/result_*.vtu'))[1:-1]
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

np.savetxt('/users/alanh/Documents/CBLResearch/lv_volumes.csv', np.array(volumes), fmt='%.2f')


# Plotting generated volume vs. time data
# (times will automatically adjust to the new, shorter length of volumes)
times = np.arange(1, len(volumes) + 1) * 0.00813

dat_times =     [0.00000, 1.61780]
dat_pressures = [0.0,     20000.0]
pressures = np.interp(times, dat_times, dat_pressures)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(volumes, pressures, markersize=3)
axs[0].set_xlabel('Volume (cm³)')
axs[0].set_ylabel('Pressure (dyne/cm²)')
axs[0].set_title('Pressure-Volume Loop')
axs[0].grid(True)

axs[1].plot(times, volumes, color='orange', markersize=3)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Volume (cm³)')
axs[1].set_title('Volume vs. Time')
axs[1].grid(True)

plt.savefig('/users/alanh/Documents/CBLResearch/pv_loop.png', dpi=300)
plt.show()