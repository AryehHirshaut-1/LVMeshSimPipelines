import numpy as np
import pyvista as pv
import tetgen
import os

from scipy.spatial import cKDTree

def create_lv_geometry(a_in, b_in, c_in, thickness, num_pts=50):
    a_out = a_in + thickness
    b_out = b_in + thickness
    c_out = c_in + thickness

    num_phi = num_pts
    theta = np.linspace(0, np.pi / 2, num_pts + 1)[1:]  # skip theta=0
    phi   = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
    T, P  = np.meshgrid(theta, phi, indexing='ij')

    def make_verts(a, b, c):
        x = a * np.sin(T) * np.cos(P)
        y = b * np.sin(T) * np.sin(P)
        z = -c * np.cos(T)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    endo_verts = make_verts(a_in,  b_in,  c_in)
    epi_verts  = make_verts(a_out, b_out, c_out)

    N          = len(endo_verts)
    epi_offset = N

    endo_apex = np.array([[0.0, 0.0, -c_in]])
    epi_apex  = np.array([[0.0, 0.0, -c_out]])

    endo_apex_idx = 2 * N
    epi_apex_idx  = 2 * N + 1

    all_verts = np.vstack([endo_verts, epi_verts, endo_apex, epi_apex])

    num_theta = len(theta)

    def grid_idx(ti, pi):
        return ti * num_phi + (pi % num_phi)

    faces      = []
    face_tags  = []   # 0=endo, 1=epi, 2=base

    # 1. ENDO surface (inward normals) — tag 0
    for ti in range(num_theta - 1):
        for pi in range(num_phi):
            a_ = grid_idx(ti,     pi)
            b_ = grid_idx(ti,     pi + 1)
            c_ = grid_idx(ti + 1, pi)
            d_ = grid_idx(ti + 1, pi + 1)
            faces += [[3, a_, c_, b_], [3, b_, c_, d_]]
            face_tags += [0, 0]

    # 2. EPI surface (outward normals) — tag 1
    for ti in range(num_theta - 1):
        for pi in range(num_phi):
            a_ = epi_offset + grid_idx(ti,     pi)
            b_ = epi_offset + grid_idx(ti,     pi + 1)
            c_ = epi_offset + grid_idx(ti + 1, pi)
            d_ = epi_offset + grid_idx(ti + 1, pi + 1)
            faces += [[3, a_, b_, c_], [3, b_, d_, c_]]
            face_tags += [1, 1]

    # 3. APEX cap — endo fan tag 0, epi fan tag 1
    for pi in range(num_phi):
        b_ = grid_idx(0, pi)
        c_ = grid_idx(0, pi + 1)
        faces += [[3, endo_apex_idx, c_, b_]]
        face_tags += [0]
        faces += [[3, epi_apex_idx, epi_offset + b_, epi_offset + c_]]
        face_tags += [1]

    # 4. BASE RING — tag 2
    for pi in range(num_phi):
        endo_i = grid_idx(num_theta - 1, pi)
        endo_j = grid_idx(num_theta - 1, pi + 1)
        epi_i  = epi_offset + grid_idx(num_theta - 1, pi)
        epi_j  = epi_offset + grid_idx(num_theta - 1, pi + 1)
        faces += [[3, endo_i, epi_i, endo_j], [3, endo_j, epi_i, epi_j]]
        face_tags += [2, 2]

    faces_np = np.array(faces, dtype=np.int_).flatten()
    surface  = pv.PolyData(all_verts, faces_np)

    surface.cell_data["region"] = np.array(face_tags, dtype=np.int32)
    surface = surface.clean(tolerance=1e-6)
    
    assert surface.is_all_triangles, "Mesh has non-triangle faces!"
    return surface

def generate_mesh_for_params(a_in, b_in, c_in, thickness, output_dir, run_id):

    """
    Takes parameters, builds the geometry, runs TetGen, and exports 
    simulation-ready files for svMultiPhysics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate boundary surface
    surface = create_lv_geometry(a_in, b_in, c_in, thickness)


    # Initialize TetGen
    tet = tetgen.TetGen(surface)
    
    # Run tetrahedralization
    # -p: PLC tetrahedralization
    # -q1.4: Enforces strict radius-edge ratio to prevent sliver elements in svMultiPhysics
    # -a0.05: Restricts maximum element volume size for convergence stability
    tet.tetrahedralize(switches="pq1.4a0.05")
    grid = tet.grid
    
    # --- BOUNDARY CONDITION TAGGING FOR SVMULTIPHYSICS ---
    # svMultiPhysics requires surface files (.vtp) to apply boundary conditions.
    # We can separate surfaces programmatically by tracking normals or radial distance.
    surf_extracted = grid.extract_surface()
    
    # Calculate distance from origin to classify inner vs outer surfaces
    centers = surf_extracted.cell_centers().points
    radial_dist = np.linalg.norm(centers[:, :2], axis=1) # Radius in XY plane
    
    # Example separation criteria based on your parametric boundaries
    endo_mask = radial_dist < (a_in + thickness/3)
    epi_mask = radial_dist > (a_in + 2*thickness/3)
    base_mask = ~(endo_mask | epi_mask)
    
    endo_vtp = surf_extracted.extract_cells(endo_mask)
    epi_vtp = surf_extracted.extract_cells(epi_mask)
    base_vtp = surf_extracted.extract_cells(base_mask)
    
    # Save files to directory
    grid.save(os.path.join(output_dir, f"mesh_{run_id}_volume.vtu")) # Master Volumetric Mesh
    endo_vtp.save(os.path.join(output_dir, f"mesh_{run_id}_endo.vtu"))
    epi_vtp.save(os.path.join(output_dir, f"mesh_{run_id}_epi.vtu"))
    base_vtp.save(os.path.join(output_dir, f"mesh_{run_id}_base.vtu"))
    
    print(f"Successfully generated case {run_id}: Tets={grid.n_cells}")

def generate_mesh_for_params1(ab_in, c_in, thickness, output_dir, run_id):
    os.makedirs(output_dir, exist_ok=True)

    surface = create_lv_geometry(ab_in, ab_in, c_in, thickness)
  
    # --- TETRAHEDRALIZE ---
    tet = tetgen.TetGen(surface)
    target_vol = (thickness / 4) ** 3 / 6
    tet.tetrahedralize(switches=f"pYq1.4a{target_vol:.4f}")
    volume = tet.grid

     # --- ASSIGN VOLUME IDs ---
    volume.point_data["GlobalNodeID"]   = np.arange(1, volume.n_points + 1, dtype=np.int32)
    volume.cell_data["GlobalElementID"] = np.arange(1, volume.n_cells  + 1, dtype=np.int32)

    vol_path = os.path.join(output_dir, f"mesh_{run_id}_volume.vtu")
    volume.save(vol_path)
    print(f"Case {run_id} — Volume: {volume.n_points} nodes, {volume.n_cells} elements")

    # --- EXTRACT SURFACE AND PROPAGATE REGION TAGS ---
    surf_extracted = volume.extract_surface()

    orig_centers = surface.cell_centers().points
    new_centers  = surf_extracted.cell_centers().points

    tree    = cKDTree(orig_centers)
    _, idxs = tree.query(new_centers)
    surf_extracted.cell_data["region"] = surface.cell_data["region"][idxs]

    # --- SANITY CHECK REGION TAGS ---
    region = surf_extracted.cell_data["region"]
    print(f"Case {run_id} — Endo: {np.sum(region==0)}, Epi: {np.sum(region==1)}, Base: {np.sum(region==2)}")

    # --- BUILD COORDINATE TREE FROM VOLUME ---
    vol_tree = cKDTree(volume.points)

    # --- SPLIT BY REGION, ASSIGN IDs, AND SAVE ---
    for tag, name in [(0, "endo"), (1, "epi"), (2, "base")]:
        surf = surf_extracted.extract_cells(region == tag).extract_geometry()

        dists, idxs = vol_tree.query(surf.points)

        if dists.max() > 1e-6:
            print(f"WARNING: case {run_id} {name} max coord mismatch = {dists.max():.2e}")

        surf.point_data["GlobalNodeID"]   = volume.point_data["GlobalNodeID"][idxs]
        surf.cell_data["GlobalElementID"] = np.arange(1, surf.n_cells + 1, dtype=np.int32)

        face_path = os.path.join(output_dir, f"mesh_{run_id}_{name}.vtp")

        surf.save(face_path)


if __name__ == "__main__":
    case_num = 0
    dir_name = f"/users/alanh/Documents/CBLResearch/lv_sim_cases/case_{case_num}"
    generate_mesh_for_params1(2.5, 10, 0.4, dir_name, case_num)

    print (f"Mesh {case_num} Generated and Fixed")