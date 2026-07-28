import pyvista as pv
import numpy as np
import glob
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

def volume_extract(file_num, general_path, mesh_path, results_path, output_path):
    ref_volume = pv.read(f'{general_path}/{mesh_path}/case_{file_num}/mesh_{file_num}_volume.vtu')
    endo_ref = pv.read(f'{general_path}/{mesh_path}/case_{file_num}/mesh_{file_num}_endo.vtp')

    ref_points = np.round(ref_volume.points, 4)
    endo_points = np.round(endo_ref.points, 4)
    endo_set = set(map(tuple, endo_points))
    endo_indices = np.array([i for i, p in enumerate(ref_points)
                            if tuple(p) in endo_set])
    print(f"[case {file_num}] Found {len(endo_indices)} endo nodes")

    files = sorted(glob.glob(f'{general_path}/{results_path}/case_{file_num}/result_NH*.vtu'))

    volumes = []
    for f in files:
        mesh = pv.read(f)
        displacement = mesh.point_data['Displacement']
        deformed = mesh.copy(deep=True)
        deformed.points = mesh.points + displacement

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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.array(volumes), fmt='%.2f')
    print(f"[case {file_num}] wrote {len(volumes)} volumes -> {output_path}")
    return volumes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, help="Case number (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--general-path", type=str, required=True)
    parser.add_argument("--mesh-path", type=str, default="MeshCasesPig")
    parser.add_argument("--results-path", type=str, default="results_HO")
    args = parser.parse_args()

    output_path = f"{args.general_path}/VolumeResults/lv_volumes_{args.case}.csv"

    volume_extract(
        file_num=args.case,
        general_path=args.general_path,
        mesh_path=args.mesh_path,
        results_path=args.results_path,
        output_path=output_path,
    )
