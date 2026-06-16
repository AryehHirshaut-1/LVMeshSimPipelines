import pyvista as pv
import numpy as np
import glob
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

# Load reference meshes to find endo node indices
ref_volume = pv.read('/users/alanh/Documents/CBLResearch/lv_sim_cases/case_0/mesh_0_volume.vtu')
endo_ref = pv.read('/users/alanh/Documents/CBLResearch/lv_sim_cases/case_0/mesh_0_endo.vtp')

ref_points = np.round(ref_volume.points, 4)
endo_points = np.round(endo_ref.points, 4)
endo_set = set(map(tuple, endo_points))
endo_indices = np.array([i for i, p in enumerate(ref_points)
                         if tuple(p) in endo_set])
print(f"Found {len(endo_indices)} endo nodes")

files = sorted(glob.glob('/users/alanh/Documents/CBLResearch/results/result_*.vtu'))
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


#Plotting generated volume vs. time data
times = np.arange(1, len(volumes) + 1) * 0.00813

dat_times =     [0.00000, 0.40650]
dat_pressures = [0.0,     20000.0]
pressures = np.interp(times, dat_times, dat_pressures)

plt.plot(volumes, pressures, marker='o', markersize=3)
plt.xlabel('Volume (cm³)')
plt.ylabel('Pressure (dyne/cm²)')
plt.title('PV Loop')
plt.grid(True)
plt.show()
