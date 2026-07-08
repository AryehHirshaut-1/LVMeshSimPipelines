import numpy as np
import pyvista as pv
import os
from scipy.spatial import cKDTree

#Create finer Volume Meshes
import gmsh
import meshio

dimensions = {
    "Radius_Lower": 2.1,
    "Radius_Higher": 2.95,
    "Height_Lower": 5,
    "Height_Higher": 12,
    "Thickness_Lower": 0.24,
    "Thickness_Higher": 0.42,
}

def generate_mesh_gmsh(ab_in, c_in, thickness, output_dir, run_id, layers_through_wall,
                        nodes_per_layer, apex_cap_frac=0.3):

    os.makedirs(output_dir, exist_ok=True)

    a_in  = ab_in
    c_out = c_in + thickness
    a_out = ab_in + thickness

    # Through-wall (radial) element size — controls layers_through_wall as before
    target_size = thickness / layers_through_wall

    # Tangential (circumferential + longitudinal) element size, derived from
    # nodes_per_layer. A "layer" here means a ring of nodes running around the
    # equator at the widest point of the epicardium (radius a_out). Arc length
    # per node on that ring becomes the target tangential element size, applied
    # uniformly across the endo/epi surfaces (it will produce somewhat tighter
    # spacing nearer the apex, where the true local radius is smaller — this is
    # expected and generally desirable for LV meshes).
    target_size_surface = (2 * np.pi * a_out) / nodes_per_layer

    gmsh.initialize()
    gmsh.model.add(f"lv_{run_id}")

    # --- Build outer (epi) and inner (endo) half-ellipsoids, then subtract ---
    def half_ellipsoid(a, c):
        vol = gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.dilate([(3, vol)], 0, 0, 0, a, a, c)
        box = gmsh.model.occ.addBox(-a - 1, -a - 1, 0, 2*(a+1), 2*(a+1), c + 1)
        result, _ = gmsh.model.occ.cut([(3, vol)], [(3, box)])
        gmsh.model.occ.synchronize()
        return result[0][1]

    outer_vol = half_ellipsoid(a_out, c_out)
    inner_vol = half_ellipsoid(a_in,  c_in)

    # Wall = outer minus inner; keep inner tool so we can tag endo surface
    wall, wall_map = gmsh.model.occ.cut(
        [(3, outer_vol)], [(3, inner_vol)],
        removeObject=True, removeTool=False
    )
    gmsh.model.occ.synchronize()

    wall_tag = wall[0][1]

    # --- Identify and tag boundary surfaces ---
    # Get all surfaces bounding the wall volume
    boundary = gmsh.model.getBoundary([(3, wall_tag)], oriented=False)
    surf_tags = [abs(b[1]) for b in boundary]

    # Classify by z-range of surface bounding box
    endo_surfs, epi_surfs, base_surfs = [], [], []
    for s in surf_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s)
        z_range = zmax - zmin
        z_top   = zmax

        # Base sits at z~0 (flat ring at the open top)
        if z_top > -1e-6 and z_range < target_size_surface * 3:
            base_surfs.append(s)
        # Endo is smaller (inner ellipsoid radius)
        elif xmax < (a_in + a_out) / 2:
            endo_surfs.append(s)
        else:
            epi_surfs.append(s)

    # Register physical groups — these are what replace the KD-tree tagging
    # Note: the epi physical group still covers the *whole* epicardium here.
    # The apex patch is split out of it later, after meshing, since carving a
    # sub-region directly out of an OCC face would require an extra
    # fragment/intersect step on the CAD geometry itself.
    pg_endo = gmsh.model.addPhysicalGroup(2, endo_surfs, tag=0)
    pg_epi  = gmsh.model.addPhysicalGroup(2, epi_surfs,  tag=1)
    pg_base = gmsh.model.addPhysicalGroup(2, base_surfs, tag=2)
    pg_wall = gmsh.model.addPhysicalGroup(3, [wall_tag], tag=10)

    gmsh.model.setPhysicalName(2, pg_endo, "endo")
    gmsh.model.setPhysicalName(2, pg_epi,  "epi")
    gmsh.model.setPhysicalName(2, pg_base, "base")
    gmsh.model.setPhysicalName(3, pg_wall, "wall")

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay — tested more uniform than Frontal-Delaunay (4)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)

    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)
    gmsh.option.setNumber("Mesh.SmoothRatio", 1.0)
    try:
        gmsh.model.mesh.generate(3)

        # Verify the mesh actually has volume elements before trusting it
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        n_tets = sum(len(t) for t in elem_tags)
        if n_tets == 0:
            raise RuntimeError("3D mesh generation produced zero tetrahedra")

        msh_path = os.path.join(output_dir, f"mesh_{run_id}.msh")
        gmsh.write(msh_path)
        gmsh.finalize()
    except Exception as e:
        print(f"CRITICAL ERROR: Mesh {run_id} failed to generate. Skipping. Error: {e}")
        gmsh.finalize()  # <-- always finalize, even on failure
        return


    # --- Convert to PyVista via meshio ---
    mio = meshio.read(msh_path)

    # Volume
    tet_cells = mio.cells_dict["tetra"]
    cell_conn  = np.hstack([np.full((len(tet_cells), 1), 4), tet_cells]).ravel()
    cell_types = np.full(len(tet_cells), pv.CellType.TETRA)
    volume = pv.UnstructuredGrid(cell_conn, cell_types, mio.points)
    volume.point_data["GlobalNodeID"]   = np.arange(1, volume.n_points + 1, dtype=np.int32)
    volume.cell_data["GlobalElementID"] = np.arange(1, volume.n_cells  + 1, dtype=np.int32)

    vol_path = os.path.join(output_dir, f"mesh_{run_id}_volume.vtu")
    volume.save(vol_path)
    print(f"Case {run_id} — Volume: {volume.n_points} nodes, {volume.n_cells} elements")

    # --- Extract and save boundary surfaces using physical group tags ---
    # meshio stores triangle sets per physical group in cells_dict keyed by tag
    vol_tree = cKDTree(mio.points)

    # Apex of the epicardium sits at (0, 0, -c_out). Triangles in the "epi"
    # physical group whose centroid falls within apex_cap_frac * a_out of
    # that point get split off into their own epi_apex patch below.
    epi_apex_point  = np.array([0.0, 0.0, -c_out])
    apex_cap_radius = apex_cap_frac * a_out

    for tag, name in [(0, "endo"), (1, "epi"), (2, "base")]:
        tri_cells = None
        for cell_block, cell_tags in zip(mio.cells, mio.cell_data.get("gmsh:physical", [[]])):
            if cell_block.type == "triangle":
                mask = cell_tags == tag
                if mask.any():
                    tri_cells = cell_block.data[mask]
                    break

        if tri_cells is None:
            print(f"WARNING: case {run_id} no triangles found for {name}")
            continue

        # Split the epicardium into the bulk patch + a small apex cap patch
        sub_groups = [(tri_cells, name)]
        if tag == 1:
            centroids  = mio.points[tri_cells].mean(axis=1)
            apex_mask  = np.linalg.norm(centroids - epi_apex_point, axis=1) < apex_cap_radius
            sub_groups = [(tri_cells[~apex_mask], "epi"), (tri_cells[apex_mask], "epi_apex")]

        for sub_tris, sub_name in sub_groups:
            if len(sub_tris) == 0:
                print(f"WARNING: case {run_id} no triangles found for {sub_name}")
                continue

            faces = np.hstack([np.full((len(sub_tris), 1), 3), sub_tris]).ravel()
            surf  = pv.PolyData(mio.points, faces)

            # Remove orphaned nodes (nodes not referenced by any face)
            surf = surf.clean(tolerance=1e-9)

            # Ensure consistent winding — flip normals to point outward for epi/base,
            # inward for endo (svMultiPhysics needs inward normals for follower pressure)
            surf.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=True)
            if sub_name == "endo":
                surf.flip_faces()  # endo normals should point inward toward the cavity

            # Map GlobalNodeID from volume by coordinate lookup
            _, idxs = vol_tree.query(surf.points)
            surf.point_data["GlobalNodeID"]   = volume.point_data["GlobalNodeID"][idxs]
            surf.cell_data["GlobalElementID"] = np.arange(1, surf.n_cells + 1, dtype=np.int32)

            # Remove the computed normals arrays before saving — svMultiPhysics computes
            # its own and will conflict with pre-stored normal arrays in the .vtp
            surf.point_data.remove("Normals")

            surf.save(os.path.join(output_dir, f"mesh_{run_id}_{sub_name}.vtp"))
            print(f"Case {run_id} — {sub_name}: {surf.n_points} nodes, {surf.n_cells} faces")


if __name__ == "__main__":

    cblresearch = os.path.join(os.path.expanduser("~"), "Documents/CBLResearch-github")

    # --- Fixed geometry: only mesh fineness varies across cases ---
    AB_IN     = 2.5   # inner equatorial radius (cm) — hold constant
    C_IN      = 8.5   # inner long-axis half-length (cm) — hold constant
    THICKNESS = 0.33   # wall thickness (cm) — hold constant
    APEX_CAP_FRAC = 0.3
    MESH_TYPE = "gmsh"   # "gmsh" or "tet"

    # --- Convergence study sweep ---
    # layers_through_wall and nodes_per_layer are both scaled by the SAME
    # refinement factor r at each step, so "fineness" is effectively a single
    # knob even though it drives two mesh parameters. Refinement is geometric
    # (roughly sqrt(2) per step) rather than linear, so cases are evenly spaced
    # on a log scale — the natural axis for plotting volume vs. mesh density.
    # Go at least one step finer than you expect to need, so the volume-vs-
    # fineness curve actually visibly flattens rather than just trending.

    #Case 3 here makes the most sense to use
    BASE_LAYERS_THROUGH_WALL = 1
    BASE_NODES_PER_LAYER     = 16
    REFINEMENT_FACTORS = [1.0, 1.5, 2.0, 2.8, 4.0, 5.6]

    fineness_values = [
        (round(BASE_LAYERS_THROUGH_WALL * r), round(BASE_NODES_PER_LAYER * r))
        for r in REFINEMENT_FACTORS
    ]

    # --- Generate one mesh per refinement level, tracking stats for convergence plotting ---
    case_stats = []

    for case_num, (layers_through_wall, nodes_per_layer) in enumerate(fineness_values):
        dir_name = os.path.join(cblresearch, f"lv_sim_cases/VolConvergence/case_{case_num}")

        if MESH_TYPE == "gmsh":
            generate_mesh_gmsh(
                AB_IN, C_IN, THICKNESS,
                dir_name, case_num,
                layers_through_wall=layers_through_wall,
                nodes_per_layer=nodes_per_layer,
                apex_cap_frac=APEX_CAP_FRAC,
            )
        elif MESH_TYPE == "tet":
            # generate_mesh_tet(...) — define this function if/when the TetGen
            # backend is needed; not implemented in this file yet.
            raise NotImplementedError("generate_mesh_tet is not defined in this script")

        vol_path = os.path.join(dir_name, f"mesh_{case_num}_volume.vtu")
        if os.path.exists(vol_path):
            mesh = pv.read(vol_path)
            case_stats.append({
                "case_num": case_num,
                "refinement_factor": REFINEMENT_FACTORS[case_num],
                "layers_through_wall": layers_through_wall,
                "nodes_per_layer": nodes_per_layer,
                "n_nodes": mesh.n_points,
                "n_elements": mesh.n_cells,
            })
        else:
            print(f"WARNING: case {case_num} produced no volume mesh — skipping in stats")

        print(f"Mesh {case_num} (layers_through_wall={layers_through_wall}, "
              f"nodes_per_layer={nodes_per_layer}) generated")

    # --- Print summary table: use this alongside cavity-volume extraction to
    # plot volume vs. node count / 1/h and check where the curve flattens ---
    print("\n--- Convergence sweep summary ---")
    print(f"{'case':>4}  {'refine':>7}  {'layers':>7}  {'nodes/layer':>11}  "
          f"{'n_nodes':>9}  {'n_elements':>10}")
    for s in case_stats:
        print(f"{s['case_num']:>4}  {s['refinement_factor']:>7.2f}  "
              f"{s['layers_through_wall']:>7}  {s['nodes_per_layer']:>11}  "
              f"{s['n_nodes']:>9}  {s['n_elements']:>10}")

    # --- Remove the intermediate .msh files (kept only .vtu/.vtp outputs) ---
    num_meshes = len(fineness_values)
    for file in range(num_meshes):
        dir_remove_name = os.path.join(cblresearch, f"lv_sim_cases/VolConvergence/case_{file}")
        msh_path = os.path.join(dir_remove_name, f"mesh_{file}.msh")
        try:
            os.remove(msh_path)
            print(f"mesh_{file}.msh removed!")
        except FileNotFoundError:
            print(f"mesh_{file}.msh already removed!")