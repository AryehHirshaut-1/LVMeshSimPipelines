""" To Do:
- Convert dimensions array to a dictionary
"""

#To Generate the Meshes
import numpy as np
import pyvista as pv
import tetgen
import os
from scipy.spatial import cKDTree

#Create finer Volume Meshes
import gmsh
import meshio

#To generate the Latin Hypercube Samples
from scipy.stats import norm
from scipy.stats import qmc


#Dimensions are Radius, Height, and Thickness (in that order), each with minimum and maxiumum values. 
#Usual Height - 5-12 cm, Usual Radius - 2.1-2.95 cm,
#Usual Thickness - 0.24-0.42 cm
dimensions = {
    "Radius_Lower": 2.1,
    "Radius_Higher": 2.95,
    "Height_Lower": 5,
    "Height_Higher": 12,
    "Thickness_Lower": 0.24,
    "Thickness_Higher": 0.42,
}

num_meshes = 100

#Choose between TetGen (less detailed walls) and Gmsh (more detailed walls, longer runtime)
type = "gmsh"


#USING TETGEN
def create_lv_geometry_tet(a_in, b_in, c_in, thickness, num_pts=50, apex_cap_frac=0.3):

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

    # Rows of the theta grid closest to the apex (ti = 0 is nearest the tip)
    # that get pulled out into the separate epi_apex patch.
    apex_band_rows = max(1, int(round(num_theta * apex_cap_frac)))

    def grid_idx(ti, pi):
        return ti * num_phi + (pi % num_phi)

    faces      = []
    face_tags  = []   # 0=endo, 1=epi, 2=base, 3=epi_apex

    # 1. ENDO surface (inward normals) — tag 0
    for ti in range(num_theta - 1):
        for pi in range(num_phi):
            a_ = grid_idx(ti,     pi)
            b_ = grid_idx(ti,     pi + 1)
            c_ = grid_idx(ti + 1, pi)
            d_ = grid_idx(ti + 1, pi + 1)
            faces += [[3, a_, c_, b_], [3, b_, c_, d_]]
            face_tags += [0, 0]

    # 2. EPI surface (outward normals) — tag 1, except the rows nearest
    #    the apex, which get tag 3 (epi_apex patch)
    for ti in range(num_theta - 1):
        epi_tag = 3 if ti < apex_band_rows else 1
        for pi in range(num_phi):
            a_ = epi_offset + grid_idx(ti,     pi)
            b_ = epi_offset + grid_idx(ti,     pi + 1)
            c_ = epi_offset + grid_idx(ti + 1, pi)
            d_ = epi_offset + grid_idx(ti + 1, pi + 1)
            faces += [[3, a_, b_, c_], [3, b_, d_, c_]]
            face_tags += [epi_tag, epi_tag]

    # 3. APEX cap — endo fan stays tag 0; epi fan joins the epi_apex patch (tag 3)
    for pi in range(num_phi):
        b_ = grid_idx(0, pi)
        c_ = grid_idx(0, pi + 1)
        faces += [[3, endo_apex_idx, c_, b_]]
        face_tags += [0]
        faces += [[3, epi_apex_idx, epi_offset + b_, epi_offset + c_]]
        face_tags += [3]

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

def generate_mesh_tet(ab_in, c_in, thickness, output_dir, run_id, apex_cap_frac=0.3):
    os.makedirs(output_dir, exist_ok=True)

    surface = create_lv_geometry_tet(ab_in, ab_in, c_in, thickness, apex_cap_frac=apex_cap_frac)
  
    #Tetrahedralize -- ONLY USE FOR create_lv_geometry, NOT create_lv_gmsh
    tet = tetgen.TetGen(surface)
    tet.tetrahedralize(switches=f"pYq1.2a0.005")
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
    print(
        f"Case {run_id} — Endo: {np.sum(region == 0)}, Epi: {np.sum(region == 1)}, "
        f"Base: {np.sum(region == 2)}, Epi apex: {np.sum(region == 3)}"
    )

    # --- BUILD COORDINATE TREE FROM VOLUME ---
    vol_tree = cKDTree(volume.points)

    # --- SPLIT BY REGION, ASSIGN IDs, AND SAVE ---
    for tag, name in [(0, "endo"), (1, "epi"), (2, "base"), (3, "epi_apex")]:
        surf = surf_extracted.extract_cells(region == tag).extract_geometry()

        dists, idxs = vol_tree.query(surf.points)

        if dists.max() > 1e-6:
            print(f"WARNING: case {run_id} {name} max coord mismatch = {dists.max():.2e}")

        surf.point_data["GlobalNodeID"]   = volume.point_data["GlobalNodeID"][idxs]
        surf.cell_data["GlobalElementID"] = np.arange(1, surf.n_cells + 1, dtype=np.int32)

        face_path = os.path.join(output_dir, f"mesh_{run_id}_{name}.vtp")

        surf.save(face_path)


#USING GMSH
#Creates Geometry and Generates Mesh
def generate_mesh_gmsh(ab_in, c_in, thickness, output_dir, run_id, layers_through_wall=3, apex_cap_frac=0.3):

    os.makedirs(output_dir, exist_ok=True)

    a_in  = ab_in
    c_out = c_in + thickness
    a_out = ab_in + thickness

    target_size = thickness / layers_through_wall

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
        if z_top > -1e-6 and z_range < target_size * 3:
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

    # --- Mesh size control ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size * 0.8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size * 2.0)
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

def lhsampler(radrangelow, radrangehigh, heightrangelow, heightrangehigh, thicknessrangelow, thicknessrangehigh):
    ranges = {
        'radius': (radrangelow, radrangehigh),     # Inner radius
        'height': (heightrangelow, heightrangehigh),     # Distance from apex to base
        'thickness': (thicknessrangelow, thicknessrangehigh)    # Myocardial wall thickness
    }

    num_samples = num_meshes
    num_vars = len(ranges)

    dist_params = {}
    for var, (vmin, vmax) in ranges.items():
        mu = (vmin + vmax) / 2.0
        sigma = (vmax - vmin) / (2 * 1.96)
        dist_params[var] = (mu, sigma)


    sampler = qmc.LatinHypercube(d=num_vars)
    lhs_samples = sampler.random(n=num_samples) 


    lv_models = np.zeros((num_samples, num_vars))
    var_names = list(ranges.keys())

    for i, var in enumerate(var_names):
        mu, sigma = dist_params[var]
        # Transform using the percent point function (inverse CDF)
        raw_samples = norm.ppf(lhs_samples[:, i], loc=mu, scale=sigma)
    
        # ADDITION: Cap the values exactly at the defined range bounds
        vmin, vmax = ranges[var]
        lv_models[:, i] = np.clip(raw_samples, a_min=vmin, a_max=vmax)

    lv_models = np.round(lv_models, decimals=3)
    return lv_models


if __name__ == "__main__":
    
    #
    cblresearch = os.path.join(os.path.expanduser("~"), "Documents/CBLResearch-github")
    
    #Perform the Latin Hypercube Sampling
    mesh_params = lhsampler(dimensions["Radius_Lower"], dimensions["Radius_Higher"], dimensions["Height_Lower"], dimensions["Height_Higher"], dimensions["Thickness_Lower"], dimensions["Thickness_Higher"])  # Example parameter ranges

    #Generates the Meshes
    for case_num, mesh_param in enumerate(mesh_params):
        dir_name = os.path.join(cblresearch, f"lv_sim_cases/case_{case_num}")

        #Change between gmsh and tet
        if type == "gmsh":
            generate_mesh_gmsh(mesh_param[0], mesh_param[1], mesh_param[2], dir_name, case_num, apex_cap_frac=apex_cap_frac)
        elif type == "tet":
            generate_mesh_tet(mesh_param[0], mesh_param[1], mesh_param[2], dir_name, case_num, apex_cap_frac=apex_cap_frac)
        print (f"Mesh {case_num} Generated and Fixed")

    #Removes the unneeded .msh files
    for file in range(0, num_meshes):
        try:
            dir_remove_name = os.path.join(cblresearch, f"lv_sim_cases/case_{file}")
            os.remove(os.path.join(dir_remove_name, f"mesh_{file}.msh"))
            print(f"mesh_{file}.msh removed!")
        except:
            print(f"mesh_{file}.msh already removed!")
