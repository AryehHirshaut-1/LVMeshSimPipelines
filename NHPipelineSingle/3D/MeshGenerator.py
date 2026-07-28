#To Generate the Meshes
import numpy as np
import pyvista as pv
import os
from scipy.spatial import cKDTree

#Create finer Volume Meshes
import gmsh
import meshio

#To generate the Latin Hypercube Samples
from scipy.stats import norm
from scipy.stats import qmc

#Create CSV with Mesh Parameters
import csv

#Dimensions are Radius, Height, and Thickness (in that order), each with minimum and maxiumum values. 
#Usual Height - 5-12 cm, Usual Radius - 2.1-2.95 cm,
#Usual Thickness - 0.24-0.42 cm
dimensions = {
    "Radius_Lower": 2.1,
    "Radius_Higher": 2.95,
    "Height_Lower": 5,
    "Height_Higher": 12,
    "Thickness_Lower": 0.6,
    "Thickness_Higher": 1.1,
}

num_meshes = 1
radius = 2.5
height = 8.5
thickness = 0.85

#General Path to Generated Mesh Directory
cblresearch = os.path.join(os.path.expanduser("~"), "HOPipelineSingle/3D")

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
            sub_groups = [(tri_cells, "epi"), (tri_cells[apex_mask], "epi_apex")]

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
    # params, case_nums = build_param_sweep(num_meshes, dimensions)
    generate_mesh_gmsh(radius, height, thickness, "HOPipelineSingle/3D")
    #Removes the unneeded .msh files
    try:
        dir_remove_name = os.path.join(cblresearch, f"case_0")
        os.remove(f"HOPipelineSingle/3D/case_0/mesh_0.msh")
        print(f"mesh_0.msh removed!")
    except:
        print(f"mesh_0.msh already removed!")
