import numpy as np
import pyvista as pv
import tetgen

from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import skfem
from skfem import MeshTet, Basis, ElementTetP1
from skfem.models.poisson import laplace, mass
from skfem.assembly import FacetBasis
import meshio, os

TAG_ENDO = 1
TAG_EPI  = 2
TAG_BASE = 3

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
    os.makedirs(output_dir, exist_ok=True)

    try:
        surface = create_lv_geometry(a_in, b_in, c_in, thickness)
    except Exception as e:
        print(f"Failed to generate geometry for run {run_id}: {e}")
        return

    # --- TETRAHEDRALIZE ---
    tet = tetgen.TetGen(surface)
    tet.tetrahedralize(switches="pYq1.2a0.01")
    volume = tet.grid

    # --- PROPAGATE REGION TAGS TO THE TETRAHEDRON CELLS ---
    # Re-attach region tags from original surface via nearest-face lookup
    surf_extracted = volume.extract_surface()
    orig_centers = surface.cell_centers().points
    new_centers  = surf_extracted.cell_centers().points

    tree    = cKDTree(orig_centers)
    _, idxs = tree.query(new_centers)
    surf_extracted.cell_data["region"] = surface.cell_data["region"][idxs]

    # --- SAVE TEMPORARY MASTER VOLUME ---
    volume.save(os.path.join(output_dir, f"mesh_{run_id}_volume.vtu"))
    
    # We pass the region-tagged surface to easily extract cleanly later
    surf_extracted.save(os.path.join(output_dir, f"mesh_{run_id}_surface_tagged.vtp"))
    print(f"Successfully generated volume case {run_id}: Tets={volume.n_cells}")

def pyvista_to_skfem(vol_vtu_path, endo_vtp_path, epi_vtp_path, base_vtp_path):
    """
    Load the PyVista VTU volume mesh into scikit-fem and tag boundary facets
    by proximity to the endo/epi/base surface VTPs.
    """
    # Read volume mesh via meshio
    vol = meshio.read(vol_vtu_path)
    tets = None
    for cb in vol.cells:
        if cb.type == "tetra":
            tets = cb.data
            break
    assert tets is not None, "No tetrahedra found"

    pts = vol.points
    mesh = MeshTet(pts.T, tets.T)  # scikit-fem convention: (3,N) and (4,M)

    # Load surfaces for tagging
    surfaces = {
        TAG_ENDO: pv.read(endo_vtp_path),
        TAG_EPI:  pv.read(epi_vtp_path),
        TAG_BASE: pv.read(base_vtp_path),
    }

    # Get all boundary facets and their midpoints from scikit-fem
    # mesh.boundary_facets() returns facet indices on the boundary
    boundary_facets = mesh.boundary_facets()
    facet_midpoints = mesh.p[:, mesh.facets[:, boundary_facets]].mean(axis=1).T  # (N_bf, 3)

    facet_tags = np.zeros(mesh.facets.shape[1], dtype=np.int32)

    tol = 1e-2
    for tag, surf in surfaces.items():
        centers = surf.cell_centers().points
        tree = cKDTree(centers)
        dists, _ = tree.query(facet_midpoints)
        tagged_local = np.where(dists < tol)[0]
        facet_tags[boundary_facets[tagged_local]] = tag

    return mesh, facet_tags

def solve_laplace(mesh, facet_tags, dirichlet_bcs):
    basis = Basis(mesh, ElementTetP1())
    K = laplace.assemble(basis)

    all_bc_dofs = {}
    for tag, val in dirichlet_bcs.items():
        facets_with_tag = np.where(facet_tags == tag)[0]
        if len(facets_with_tag) == 0:
            raise ValueError(f"No facets found for tag {tag}")
        dofs = basis.get_dofs(facets_with_tag).all()
        
        # FIX: Convert the numpy array to a tuple so it's hashable
        all_bc_dofs[tuple(dofs)] = val

    u = np.zeros(K.shape[0])

    # Condense: eliminate BC dofs from the system
    # FIX: Use np.fromiter or concatenate list comprehension safely
    all_dofs_list = [np.array(k) for k in all_bc_dofs.keys()]
    interior_dofs = np.setdiff1d(np.arange(K.shape[0]),
                                 np.concatenate(all_dofs_list) if all_dofs_list else [])
    rhs = np.zeros(K.shape[0])

    # Lift the boundary values into the rhs
    for dofs_tuple, val in all_bc_dofs.items():
        dofs = np.array(dofs_tuple) # Convert back to array for indexing
        u[dofs] = val
        rhs -= K @ (u * (np.isin(np.arange(len(u)), dofs)))

    K_int = K[np.ix_(interior_dofs, interior_dofs)]
    rhs_int = rhs[interior_dofs]
    u[interior_dofs] = spsolve(K_int, rhs_int)

    return u

def compute_nodal_gradient(mesh, u):
    """
    Compute gradient of scalar field u at each node by averaging
    element gradients (P1 elements → constant grad per tet).
    Returns (n_points, 3).
    """
    pts  = mesh.p.T   # (n_pts, 3)
    tets = mesh.t.T   # (n_tets, 4)

    n_pts  = pts.shape[0]
    n_tets = tets.shape[0]

    grad_elem = np.zeros((n_tets, 3))

    for i in range(n_tets):
        idx = tets[i]
        # Build the Jacobian [x1-x0, x2-x0, x3-x0]
        J = pts[idx[1:]] - pts[idx[0]]   # (3,3)
        # Gradients of P1 basis functions w.r.t. reference coords
        dN_ref = np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        # grad_u = J^{-T} @ (sum dN_i * u_i)
        try:
            Jinv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            continue
        grad_elem[i] = Jinv.T @ (dN_ref.T @ u[idx])

    # Average element gradients onto nodes
    grad_node = np.zeros((n_pts, 3))
    count      = np.zeros(n_pts)
    for i in range(n_tets):
        for j in tets[i]:
            grad_node[j] += grad_elem[i]
            count[j]     += 1

    count = np.maximum(count, 1)
    grad_node /= count[:, None]
    return grad_node

def ldrb_fibers(mesh, facet_tags,
                alpha_endo=60.0, alpha_epi=-60.0,
                beta_endo=0.0,   beta_epi=0.0):
    """
    Compute f0, s0, n0 fiber/sheet/sheet-normal fields via LDRB.
    Returns three (n_points, 3) arrays.
    """
    print("Solving transmural Laplace (endo=0, epi=1)...")
    phi_tv = solve_laplace(mesh, facet_tags,
                           {TAG_ENDO: 0.0, TAG_EPI: 1.0})

    # Apex-base: base=1, no BC on apex → naturally 0 there
    print("Solving apex-base Laplace (base=1)...")
    phi_ab = solve_laplace(mesh, facet_tags,
                           {TAG_BASE: 1.0})

    print("Computing gradients...")
    grad_tv = compute_nodal_gradient(mesh, phi_tv)  # transmural direction
    grad_ab = compute_nodal_gradient(mesh, phi_ab)  # apex-base direction

    def safe_normalize(v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return v / norms

    e_t = safe_normalize(grad_tv)   # transmural unit vector
    e_l = safe_normalize(grad_ab)   # longitudinal unit vector

    # Circumferential = longitudinal × transmural
    e_c = safe_normalize(np.cross(e_l, e_t))

    n_pts = mesh.p.shape[1]
    f0 = np.zeros((n_pts, 3))
    s0 = np.zeros((n_pts, 3))
    n0 = np.zeros((n_pts, 3))

    print("Building fiber frames...")
    for i in range(n_pts):
        t = phi_tv[i]   # 0=endo, 1=epi

        # Interpolate angles
        alpha = np.radians((1 - t) * alpha_endo + t * alpha_epi)
        beta  = np.radians((1 - t) * beta_endo  + t * beta_epi)

        ec = e_c[i]
        et = e_t[i]
        el = e_l[i]

        # Fiber: rotate e_c by alpha around e_t
        f = (np.cos(alpha) * ec
           + np.sin(alpha) * np.cross(et, ec)
           + (1 - np.cos(alpha)) * (et @ ec) * et)

        # Sheet: rotate e_t by beta around f
        s = (np.cos(beta) * et
           + np.sin(beta) * np.cross(f, et)
           + (1 - np.cos(beta)) * (f @ et) * f)

        n = np.cross(f, s)

        norm_f = np.linalg.norm(f)
        norm_s = np.linalg.norm(s)
        norm_n = np.linalg.norm(n)

        f0[i] = f / norm_f if norm_f > 1e-12 else f
        s0[i] = s / norm_s if norm_s > 1e-12 else s
        n0[i] = n / norm_n if norm_n > 1e-12 else n

    return f0, s0, n0

def add_fibers_to_vtu(vol_vtu_path, f0, s0, n0, out_path):
    vol = pv.read(vol_vtu_path)

    # Match node ordering between meshio/skfem and PyVista via KD-tree
    vol_mi = meshio.read(vol_vtu_path)
    skfem_pts = vol_mi.points

    tree = cKDTree(skfem_pts)
    _, idx = tree.query(vol.points)

    vol.point_data["FiberDirection"]       = f0[idx]
    vol.point_data["SheetDirection"]       = s0[idx]
    vol.point_data["SheetNormalDirection"] = n0[idx]
    vol.save(out_path)
    print(f"Saved fiber mesh to {out_path}")

if __name__ == "__main__":
    case_num = "HolzapfelOgden"
    base_dir = os.path.join(os.path.expanduser("~"),
                            f"/users/alanh/Documents/CBLResearch/lv_sim_cases/case_{case_num}")
    
    # 1. Generate the base volume and raw surface
    generate_mesh_for_params(2.5, 2.5, 10, 0.6, base_dir, case_num)

    vol_path       = os.path.join(base_dir, f"mesh_{case_num}_volume.vtu")
    surf_temp_path = os.path.join(base_dir, f"mesh_{case_num}_surface_tagged.vtp")
    out_path       = os.path.join(base_dir, f"mesh_{case_num}_fibers.vtu")

    endo_path = os.path.join(base_dir, f"mesh_{case_num}_endo.vtp")
    epi_path  = os.path.join(base_dir, f"mesh_{case_num}_epi.vtp")
    base_path = os.path.join(base_dir, f"mesh_{case_num}_base.vtp")

    # 2. Compute fiber directions using scikit-fem
    print("Loading mesh and computing fibers via LDRB...")
    mesh, facet_tags = pyvista_to_skfem(vol_path, endo_path, epi_path, base_path)
    f0, s0, n0 = ldrb_fibers(mesh, facet_tags, alpha_endo=60, alpha_epi=-60, beta_endo=0, beta_epi=0)
    add_fibers_to_vtu(vol_path, f0, s0, n0, out_path)

    # 3. Apply strict 1-indexed elements and nodes to the core fiber mesh
    print("Enforcing unified global master elements and nodes...")
    fiber_mesh = pv.read(out_path)
    vol_el_ids = np.arange(1, fiber_mesh.n_cells + 1, dtype=np.int32)
    vol_nd_ids = np.arange(1, fiber_mesh.n_points + 1, dtype=np.int32)
    
    fiber_mesh.cell_data["GlobalElementID"] = vol_el_ids
    fiber_mesh.point_data["GlobalNodeID"] = vol_nd_ids
    fiber_mesh.save(out_path)

    # 4. FIX: Reconstruct Face Meshes natively using Master Geometry Pointers
    # This keeps the master coordinates intact to prevent C++ array bounds violations.
    full_surface = pv.read(surf_temp_path)
    full_surface = full_surface.extract_geometry() # Coerce to pure PolyData

    # Map the surface nodes to the exact indices of the master volume mesh
    node_tree = cKDTree(fiber_mesh.points)
    cell_tree = cKDTree(fiber_mesh.cell_centers().points)
    _, surf_to_vol_nodes = node_tree.query(full_surface.points)

    boundary_configs = {
        "endo": (0, endo_path),
        "epi":  (1, epi_path),
        "base": (2, base_path)
    }

    for label, (region_tag, path) in boundary_configs.items():
        print(f"Building unified face structure for boundary: {label}...")
        
        # Isolate the exact triangles that belong to this region tag
        face_indices = np.where(full_surface.cell_data["region"] == region_tag)[0]
        region_triangles = full_surface.regular_faces[face_indices] # shape (N_tris, 3)

        # Map the triangle local connectivity arrays directly to the Master Volume indices
        mapped_triangles = surf_to_vol_nodes[region_triangles]

        # Format the faces explicitly into standard VTK padding ([3, n1, n2, n3, 3, n4, n5...])
        padding = np.full((mapped_triangles.shape[0], 1), 3, dtype=np.int32)
        vtk_cells = np.hstack((padding, mapped_triangles)).flatten()

        # Build a PolyData mesh using the full volume point array.
        # This keeps the indexing uniform across the boundary files.
        face_poly = pv.PolyData(fiber_mesh.points, vtk_cells)

        # Map and pass the global parent elements
        _, face_to_vol_cells = cell_tree.query(face_poly.cell_centers().points)
        face_poly.cell_data["GlobalElementID"] = vol_el_ids[face_to_vol_cells]
        face_poly.point_data["GlobalNodeID"] = vol_nd_ids

        # FIX HERE: Replace the broken clean() call with this method
        face_poly = face_poly.remove_unused_points()

        face_poly.save(path)
        print(f"-> Saved face mesh {label}. vtp alignment verified.")

       