import glob
import os
import re
import warnings
import numpy as np
import pyvista as pv

warnings.filterwarnings('ignore', category=pv.PyVistaFutureWarning)

BASE = "/home/ah2699/CBLResearch-github/VolumeExtractions"

# ============ CONFIG — edit if needed, then click Run ============
MATERIAL_TYPE = "HO"
OUT_DIR = f"{BASE}/VolumesCSV"
# ===================================================================


def discover_cases(base):
    """Scan <base>/lv_sim_cases/case_* and return sorted list of case numbers."""
    case_dirs = glob.glob(f'{base}/lv_sim_cases/case_*')
    case_nums = []
    for d in case_dirs:
        if not os.path.isdir(d):
            continue
        m = re.search(r'case_(\d+)$', d)
        if m:
            case_nums.append(int(m.group(1)))
    return sorted(case_nums)


def volume_extract(file_num, material_type, out_dir):
    # Load reference meshes to find endo node indices
    ref_volume = pv.read(f'{BASE}/lv_sim_cases/case_{file_num}/mesh_{file_num}_volume.vtu')
    endo_ref = pv.read(f'{BASE}/lv_sim_cases/case_{file_num}/mesh_{file_num}_endo.vtp')

    ref_points = np.round(ref_volume.points, 4)
    endo_points = np.round(endo_ref.points, 4)
    endo_set = set(map(tuple, endo_points))
    endo_indices = np.array([i for i, p in enumerate(ref_points)
                             if tuple(p) in endo_set])
    print(f"[case {file_num}] Found {len(endo_indices)} endo nodes", flush=True)

    # Include all files (first and last included) -> 200 data points
    files = sorted(glob.glob(
        f'{BASE}/results_{material_type}/case_{file_num}/result_{material_type}_*.vtu'))

    if not files:
        raise FileNotFoundError(
            f"No result files found for case {file_num} "
            f"(material_type={material_type})")

    volumes = []
    endo_index_set = set(endo_indices.tolist())
    for f in files:
        mesh = pv.read(f)
        displacement = mesh.point_data['Displacement']
        deformed = mesh.copy(deep=True)
        deformed.points = mesh.points + displacement

        surface = deformed.extract_surface(algorithm='dataset_surface')
        orig_ids = surface.point_data['vtkOriginalPointIds']
        surface_endo_mask = np.array([pid in endo_index_set for pid in orig_ids])

        endo_surf = surface.extract_points(surface_endo_mask, adjacent_cells=True)
        endo_surf = endo_surf.extract_surface()
        endo_surf = endo_surf.triangulate()
        endo_closed = endo_surf.fill_holes(1000)
        endo_closed = endo_closed.triangulate()

        volumes.append(endo_closed.volume)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'case_{file_num}.csv')
    np.savetxt(out_path, np.array(volumes), fmt='%.2f')
    print(f"[case {file_num}] wrote {len(volumes)} volumes -> {out_path}", flush=True)
    return volumes


def run_sweep(material_type, out_dir):
    cases = discover_cases(BASE)
    if not cases:
        print(f"No case directories found under {BASE}/lv_sim_cases/", flush=True)
        return

    print(f"Discovered {len(cases)} case(s): {cases}", flush=True)

    succeeded = []
    failed = []
    for case_num in cases:
        try:
            volume_extract(case_num, material_type, out_dir)
            succeeded.append(case_num)
        except Exception as e:
            print(f"[case {case_num}] WARNING: skipped due to error: {e}", flush=True)
            failed.append((case_num, str(e)))

    print("\n===== Sweep summary =====", flush=True)
    print(f"Succeeded ({len(succeeded)}): {succeeded}", flush=True)
    if failed:
        print(f"Failed ({len(failed)}):", flush=True)
        for case_num, err in failed:
            print(f"  case {case_num}: {err}", flush=True)
    else:
        print("Failed (0): none", flush=True)


if __name__ == "__main__":
    run_sweep(MATERIAL_TYPE, OUT_DIR)