import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings

# ============================================================
# Helpers (mesh I/O + distance maps)
# ============================================================
def ensure_polydata(mesh):
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.clean()


def split_two_surfaces(mesh: pv.PolyData):
    
    conn = mesh.connectivity()
    ids = conn["RegionId"]
    unique, counts = np.unique(ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    if len(order) < 2:
        raise RuntimeError("Found <2 connected components. STL must contain exactly two surfaces.")
    reg0 = unique[order[0]]
    reg1 = unique[order[1]]
    s0 = conn.threshold([reg0, reg0], scalars="RegionId") \
     .extract_surface(algorithm="dataset_surface").clean()
    s1 = conn.threshold([reg1, reg1], scalars="RegionId") \
     .extract_surface(algorithm="dataset_surface").clean()

    return s0, s1


def pick_inner_outer_by_radius(s0: pv.PolyData, s1: pv.PolyData):
    c = np.vstack([s0.points, s1.points]).mean(axis=0)
    r0 = np.mean(np.linalg.norm(s0.points - c, axis=1))
    r1 = np.mean(np.linalg.norm(s1.points - c, axis=1))
    return (s1, s0) if r0 > r1 else (s0, s1)  # inner, outer


def compute_map_nn(a: pv.PolyData, b: pv.PolyData, name="dist"):
    """Nearest-vertex distance a->b, and duplicate to b from nearest a."""
    a_pts = a.points
    b_pts = b.points

    b_tree = KDTree(b_pts)
    d_a, _ = b_tree.query(a_pts, k=1)

    a_out = a.copy()
    a_out[name] = d_a.astype(np.float64)

    a_tree = KDTree(a_pts)
    _, idx_a = a_tree.query(b_pts, k=1)

    b_out = b.copy()
    b_out[name] = a_out[name][idx_a].astype(np.float64)
    return a_out, b_out


def pca_alignment(source: pv.PolyData, target: pv.PolyData) -> np.ndarray:
    src_centered = source.points - np.mean(source.points, axis=0)
    tgt_centered = target.points - np.mean(target.points, axis=0)

    cov_src = np.cov(src_centered, rowvar=False)
    cov_tgt = np.cov(tgt_centered, rowvar=False)

    _, eigvec_src = np.linalg.eigh(cov_src)
    _, eigvec_tgt = np.linalg.eigh(cov_tgt)

    eigvec_src[:, -1] *= np.sign(np.linalg.det(eigvec_src))
    eigvec_tgt[:, -1] *= np.sign(np.linalg.det(eigvec_tgt))

    R = eigvec_tgt @ eigvec_src.T
    aligned_points = (src_centered @ R.T) + np.mean(target.points, axis=0)
    return aligned_points


def align_and_trim_surfaces(eroded: pv.PolyData,
                            reference: pv.PolyData,
                            scaling_factor=1.0,
                            offset_x=0.0, offset_y=0.0, offset_z=0.0,
                            do_trim=True):
    """PCA align + offsets + optional scale + optional trim protrusions."""
    ref = reference.copy(deep=True)
    ref.points = pca_alignment(ref, eroded)
    ref.points += np.array([offset_x, offset_y, offset_z], dtype=float)

    c = np.mean(ref.points, axis=0)
    ref.points = c + float(scaling_factor) * (ref.points - c)

    if do_trim:
        er = eroded.copy()
        er.compute_normals(inplace=True, consistent_normals=True)
        ref.compute_normals(inplace=True, consistent_normals=True)

        tree = KDTree(er.points)
        _, idx = tree.query(ref.points, k=1)
        vectors = er.points[idx] - ref.points
        signed_dists = np.sum(vectors * ref.point_normals, axis=1)
        trim_mask = signed_dists > 0
        ref.points[trim_mask] = er.points[idx[trim_mask]]

    return ref


def distance_eroded_to_recon_unsigned(eroded_inner: pv.PolyData, recon: pv.PolyData):
    """
    Unsigned point-to-surface distance defined on ERODED surface.
    Stored as 'erosion' and alias 'sep_dist'.
    """
    recon_tri = recon.triangulate().clean()
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(recon_tri)

    pts = np.asarray(eroded_inner.points)
    d = np.empty(len(pts), dtype=float)
    for i, p in enumerate(pts):
        d[i] = abs(imp.EvaluateFunction(p))

    out = eroded_inner.copy()
    out["erosion"] = d.astype(np.float64)
    out["sep_dist"] = out["erosion"]
    return out


def box_dims_from_bounds(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (xmax - xmin), (ymax - ymin), (zmax - zmin)


def vtk_interactor(plotter: pv.Plotter):
    try:
        return plotter.iren.interactor
    except Exception:
        pass
    try:
        return plotter.iren._iren
    except Exception:
        pass
    raise RuntimeError("Cannot access VTK interactor.")


def get_scalar_bar_actor(plotter: pv.Plotter):
    actors2d = plotter.renderer.GetActors2D()
    actors2d.InitTraversal()
    for _ in range(actors2d.GetNumberOfItems()):
        a = actors2d.GetNextActor2D()
        if isinstance(a, vtk.vtkScalarBarActor):
            return a
    return None


def apply_scalarbar_layout(sb: vtk.vtkScalarBarActor, vertical: bool):
    pos = sb.GetPositionCoordinate()
    pos.SetCoordinateSystemToNormalizedViewport()
    if vertical:
        sb.SetOrientationToVertical()
        pos.SetValue(0.88, 0.12)
        sb.SetWidth(0.10)
        sb.SetHeight(0.78)
    else:
        sb.SetOrientationToHorizontal()
        pos.SetValue(0.20, 0.05)
        sb.SetWidth(0.60)
        sb.SetHeight(0.08)


# ============================================================
# Geodesic zoning (parallel to intrados boundary)
# ============================================================
def build_edge_graph(mesh: pv.PolyData):
    """Return CSR adjacency matrix (N x N) with edge weights = Euclidean edge length."""
    tri = mesh.triangulate().clean()
    faces = tri.faces.reshape(-1, 4)[:, 1:]  # (n_tri, 3)
    pts = tri.points
    n = tri.n_points

    # Collect unique edges from triangles
    edges = set()
    for a, b, c in faces:
        edges.add((int(a), int(b)))
        edges.add((int(b), int(c)))
        edges.add((int(c), int(a)))
        edges.add((int(b), int(a)))
        edges.add((int(c), int(b)))
        edges.add((int(a), int(c)))

    ii = []
    jj = []
    ww = []
    for i, j in edges:
        ii.append(i)
        jj.append(j)
        ww.append(float(np.linalg.norm(pts[i] - pts[j])))

    A = coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
    return tri, A


def compute_geodesic_zones_from_intrados(mesh_eroded: pv.PolyData,
                                        erosion: np.ndarray,
                                        noise_thr_mm: float = 0.12,
                                        n_zones: int = 6):
    """
    Zone 0 = near-zero erosion (<= noise threshold).
    Remaining surface split into n_zones bands by geodesic distance from Zone-0 boundary.
    Returns:
        mesh_tri (triangulated),
        zone_id (N,),
        dist_geo (N,)
    """
    mesh_tri, A = build_edge_graph(mesh_eroded)
    er = np.asarray(erosion, float)
    N = mesh_tri.n_points

    if er.shape[0] != N:
        # erosion was on original mesh; triangulate preserves points, but be safe:
        # nearest mapping from original points to tri points
        tree = KDTree(mesh_eroded.points)
        _, idx = tree.query(mesh_tri.points, k=1)
        er = er[idx]

    zone0 = (er <= noise_thr_mm)

    # Boundary = nodes NOT in zone0 that touch zone0 by an edge
    # We can find neighbors via adjacency structure
    boundary = np.zeros(N, dtype=bool)
    A_csr = A.tocsr()
    for i in np.where(~zone0)[0]:
        nbrs = A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]]
        if np.any(zone0[nbrs]):
            boundary[i] = True

    boundary_idx = np.where(boundary)[0]
    if boundary_idx.size < 10:
        # fallback: if boundary is tiny, use all zone0 as sources (still works)
        boundary_idx = np.where(zone0)[0]
    if boundary_idx.size < 10:
        raise RuntimeError("Could not form intrados boundary sources for geodesic zones.")

    # Multi-source Dijkstra:
    # create a super-source node connected to all boundary nodes with 0 weight
    super_node = N
    ii = [super_node] * boundary_idx.size
    jj = boundary_idx.tolist()
    ww = [0.0] * boundary_idx.size

    # Also connect boundary nodes back to super node (not strictly needed)
    ii += boundary_idx.tolist()
    jj += [super_node] * boundary_idx.size
    ww += [0.0] * boundary_idx.size

    A_aug = csr_matrix(
        (np.hstack([A.data, np.array(ww)]),
         (np.hstack([A.nonzero()[0], np.array(ii)]),
          np.hstack([A.nonzero()[1], np.array(jj)]))),
        shape=(N + 1, N + 1)
    )
    # dijkstra from super_node to all
    dist_all = dijkstra(A_aug, directed=True, indices=super_node)
    dist_geo = np.asarray(dist_all[:N], float)

    # Build zone ids
    zone_id = np.zeros(N, dtype=int)
    zone_id[zone0] = 0

    mask = ~zone0
    if np.any(mask):
        dmax = float(np.nanmax(dist_geo[mask]))
        if dmax <= 1e-12:
            # everything is zone0-ish
            zone_id[mask] = 1
        else:
            delta = dmax / float(n_zones)
            zid = 1 + np.floor(dist_geo[mask] / delta).astype(int)
            zid = np.clip(zid, 1, n_zones)
            zone_id[mask] = zid

    return mesh_tri, zone_id, dist_geo


def make_zone_boundary_lines(mesh_tri: pv.PolyData, zone_id: np.ndarray):
    """
    Make polyline boundaries between integer zones by contouring half-integers.
    """
    out = mesh_tri.copy()
    out["zone_id"] = zone_id.astype(np.float64)
    maxz = int(np.max(zone_id))
    isos = [k + 0.5 for k in range(0, maxz)]  # 0.5,1.5,2.5,...
    if not isos:
        return None
    lines = out.contour(isosurfaces=isos, scalars="zone_id")
    return lines


def zone_label_positions(mesh_tri: pv.PolyData, zone_id: np.ndarray, dist_geo: np.ndarray):
    """
    Put label at an actual vertex on the surface:
      - For zone 0: pick a vertex in zone0 near its "center" (median of points)
      - For others: pick vertex with dist_geo closest to zone's median dist
    Returns list of (pos, label_str)
    """
    pts = mesh_tri.points
    labels = []
    maxz = int(np.max(zone_id))

    for z in range(0, maxz + 1):
        idx = np.where(zone_id == z)[0]
        if idx.size < 50:
            continue

        if z == 0:
            # pick vertex closest to centroid of zone0 points
            c = pts[idx].mean(axis=0)
            j = idx[np.argmin(np.linalg.norm(pts[idx] - c, axis=1))]
            labels.append((pts[j], f"Z{z}"))
        else:
            d = dist_geo[idx]
            dmed = float(np.median(d))
            j = idx[np.argmin(np.abs(d - dmed))]
            labels.append((pts[j], f"Z{z}"))

    return labels


def fit_candidates(data):
    candidates = {
        "Normal": stats.norm,
        "Exponential": stats.expon,
        "Gamma": stats.gamma,
        "Lognormal": stats.lognorm,
        "Weibull": stats.weibull_min,
    }

    results = {}
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]

    if data.size < 30:
        return None

    for name, dist in candidates.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                params = dist.fit(data)
                ks_stat, p_val = stats.kstest(data, dist.name, args=params)
            results[name] = (ks_stat, p_val, params, dist)
        except Exception:
            continue

    if not results:
        return None

    best_name, (ks, pval, params, dist) = sorted(results.items(), key=lambda x: x[1][0])[0]
    return best_name, ks, pval, params, dist, results

def normality_report(data):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]

    if data.size < 8:
        return np.nan, np.nan, "insufficient_data"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat, pval = stats.normaltest(data)
        return float(stat), float(pval), "normaltest"
    except Exception:
        return np.nan, np.nan, "failed"


# ============================================================
# Bimodality checks for Zone 1
# ============================================================
def kde_peak_count(x, bw_method="scott", grid_n=400):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        return 0, None, None
    kde = gaussian_kde(x, bw_method=bw_method)
    xx = np.linspace(x.min(), x.max(), grid_n)
    yy = kde(xx)
    peaks, _ = find_peaks(yy)
    return int(len(peaks)), xx, yy


def bimodality_coefficient(x):
    """
    BC = (skew^2 + 1) / (kurtosis + correction)
    Heuristic: BC > 0.555 suggests bimodality (not a proof).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 25:
        return np.nan
    g = stats.skew(x, bias=False)
    k = stats.kurtosis(x, fisher=False, bias=False)  # Pearson kurtosis
    corr = 3.0 * (n - 1)**2 / ((n - 2) * (n - 3)) if n > 3 else 0.0
    bc = (g**2 + 1.0) / (k + corr)
    return float(bc)


def zone1_bimodality_report(erosion_depths, zone_id, noise_thr=0.12):
    z = (zone_id == 1)
    x_raw = np.asarray(erosion_depths[z], float)
    x_raw = x_raw[np.isfinite(x_raw)]
    rep = {
        "n_raw": int(x_raw.size),
        "kde_peaks_raw": None,
        "BC_raw": None,
        "n_filtered": None,
        "kde_peaks_filtered": None,
        "BC_filtered": None,
        "dip_pvalue": None,
        "dip_stat": None,
    }
    if x_raw.size < 30:
        return rep

    rep["kde_peaks_raw"], _, _ = kde_peak_count(x_raw)
    rep["BC_raw"] = bimodality_coefficient(x_raw)

    x = x_raw[x_raw >= noise_thr]
    rep["n_filtered"] = int(x.size)
    if x.size >= 30:
        rep["kde_peaks_filtered"], _, _ = kde_peak_count(x)
        rep["BC_filtered"] = bimodality_coefficient(x)

        # Optional dip test
        try:
            import diptest
            dip, pval = diptest.diptest(x)
            rep["dip_stat"] = float(dip)
            rep["dip_pvalue"] = float(pval)
        except Exception:
            pass

    return rep


# ============================================================
# USER SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TWO_WALLS_STL = BASE_DIR / "data" / "cut_initial_thickness_1_7_mm.stl"
RECON_STL     = BASE_DIR / "data" / "New_reconstructed_1_7.stl"

RESULTS_DIR = BASE_DIR / "results"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"

THICK_CLIM   = (0.0, 13.0)
EROSION_CLIM = (0.0, 4.5)

THICK_CMAP   = "viridis"
EROSION_CMAP = "inferno"

OUTER_OPACITY = 1.0
RECON_OPACITY = 0

SAMPLE_LABEL_FONT_SIZE = 15
SAMPLE_LABEL_POINT_SIZE = 10
SAMPLE_LABEL_SHAPE_OPACITY = 0.35

# Recon alignment controls (same style as your erosion code)
SCALING_FACTOR = 1.0
OFFSET_X = -2.0
OFFSET_Y = -2.0
OFFSET_Z = 0.12
DO_TRIM = True

# Erosion noise threshold
NOISE_THRESHOLD_MM = 0.12

# Zoning
N_ZONES = 4  # total nonzero bands beyond Zone 0


# ============================================================
# Load + shift-to-origin
# ============================================================

if not TWO_WALLS_STL.exists():
    raise FileNotFoundError(f"Two-wall STL not found: {TWO_WALLS_STL}")
if not RECON_STL.exists():
    raise FileNotFoundError(f"Reconstructed STL not found: {RECON_STL}")

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

two = ensure_polydata(pv.read(str(TWO_WALLS_STL)))


shift = np.array(two.center, dtype=float)

two0 = two.copy()
two0.points = two0.points - shift

s0, s1 = split_two_surfaces(two0)
eroded_inner0, uneroded_outer0 = pick_inner_outer_by_radius(s0, s1)

# measured thickness
inner_meas, outer_meas = compute_map_nn(eroded_inner0, uneroded_outer0, name="thickness")

# recon align

recon_raw = ensure_polydata(pv.read(str(RECON_STL)))

recon0 = recon_raw.copy()
recon0.points = recon0.points - shift

recon_aligned = align_and_trim_surfaces(
    eroded=eroded_inner0,
    reference=recon0,
    scaling_factor=SCALING_FACTOR,
    offset_x=OFFSET_X, offset_y=OFFSET_Y, offset_z=OFFSET_Z,
    do_trim=DO_TRIM
)

# recon thickness (recon->outer)
recon_thick, outer_recon_thick = compute_map_nn(recon_aligned, uneroded_outer0, name="thickness_recon")

# erosion/separation distance map on eroded inner (defined on eroded)
inner_eros = distance_eroded_to_recon_unsigned(eroded_inner0, recon_aligned)

# Build zones on the eroded surface (for view 4)
mesh_zones_tri, zone_id, dist_geo = compute_geodesic_zones_from_intrados(
    mesh_eroded=inner_eros,  # use eroded surface geometry
    erosion=np.asarray(inner_eros["erosion"]),
    noise_thr_mm=NOISE_THRESHOLD_MM,
    n_zones=N_ZONES
)

# attach zone id back onto inner_eros (need mapping if triangulation changed)
# Here triangulate preserves points; we keep it safe with nearest map.
tree_eros = KDTree(mesh_zones_tri.points)
_, idx_map = tree_eros.query(inner_eros.points, k=1)
inner_eros = inner_eros.copy()
inner_eros["zone_id"] = zone_id[idx_map].astype(np.int32)

zone_lines = make_zone_boundary_lines(mesh_zones_tri, zone_id)
zone_labels = zone_label_positions(mesh_zones_tri, zone_id, dist_geo)

# Bimodality report for Zone 1
rep = zone1_bimodality_report(np.asarray(inner_eros["erosion"]), np.asarray(inner_eros["zone_id"]), NOISE_THRESHOLD_MM)
z1_all = np.asarray(inner_eros["erosion"])[np.asarray(inner_eros["zone_id"]) == 1]
z1_all = z1_all[np.isfinite(z1_all)]
z1_norm_stat, z1_norm_pval, z1_norm_method = normality_report(z1_all)

rep["normality_method"] = z1_norm_method
rep["normality_stat"] = z1_norm_stat
rep["normality_pvalue"] = z1_norm_pval


print("\n=== Zone 1 bimodality check ===")
for k, v in rep.items():
    print(f"{k:18s}: {v}")

# box dims
dx, dy, dz = box_dims_from_bounds(two0.bounds)
dims_text = f"Box ΔX={dx:.2f} mm  ΔY={dy:.2f} mm  ΔZ={dz:.2f} mm"


# ============================================================
# Plot + UI state
# ============================================================
p = pv.Plotter()
p.set_background("white")

try:
    p.ren_win.StereoCapableWindowOff()
    p.ren_win.SetStereoRender(False)
except Exception:
    pass

state = {
    "view": 1,                 # 1 thickness, 2 recon thickness, 3 erosion, 4 erosion+zones
    "show_outer": True,
    "show_box": True,
    "scalar_vertical": True,
}

inner_actor = None
outer_actor = None
recon_actor = None
bbox_actor = None
bounds_actor = None
zone_lines_actor = None
zone_label_actors = []

picker = vtk.vtkCellPicker()
picker.SetTolerance(0.0005)
picker.PickFromListOn()
_current_pick_mesh = None
_current_pick_scalar = None

_samples = []


def remove_actor_safe(name):
    try:
        p.remove_actor(name)
    except Exception:
        pass


def clear_zone_actors():
    global zone_lines_actor, zone_label_actors
    if zone_lines_actor is not None:
        try:
            p.remove_actor("ZONE_LINES")
        except Exception:
            pass
        zone_lines_actor = None
    for nm in list(zone_label_actors):
        try:
            p.remove_actor(nm)
        except Exception:
            pass
    zone_label_actors = []


def rebuild_scene():
    global inner_actor, outer_actor, recon_actor, bbox_actor, bounds_actor
    global _current_pick_mesh, _current_pick_scalar, zone_lines_actor, zone_label_actors

    for nm in ["INNER", "OUTER", "RECON", "BBOX", "BOUNDS", "HUD"]:
        remove_actor_safe(nm)
    clear_zone_actors()

    # ---------------- View 1 ----------------
    if state["view"] == 1:
        inner_actor = p.add_mesh(
            inner_meas, scalars="thickness", clim=THICK_CLIM, cmap=THICK_CMAP,
            show_edges=False, show_scalar_bar=True,
            scalar_bar_args={"title": "Thickness (mm)", "n_labels": 6},
            culling=False, name="INNER"
        )
        outer_actor = p.add_mesh(
            outer_meas, scalars="thickness", clim=THICK_CLIM, cmap=THICK_CMAP,
            opacity=OUTER_OPACITY, show_edges=False, show_scalar_bar=False,
            culling=False, name="OUTER"
        )
        recon_actor = p.add_mesh(
            recon_thick, color="lightgray", opacity=RECON_OPACITY,
            show_edges=True, line_width=1, show_scalar_bar=False,
            culling=False, name="RECON"
        )
        _current_pick_mesh = inner_meas
        _current_pick_scalar = "thickness"
        picker.InitializePickList()
        picker.PickFromListOn()
        picker.AddPickList(inner_actor)

    # ---------------- View 2 ----------------
    elif state["view"] == 2:
        # Hide eroded inner completely
        inner_actor = p.add_mesh(
            inner_meas, color="white", opacity=0.0,
            show_edges=False, show_scalar_bar=False,
            culling=False, name="INNER"
        )
        inner_actor.SetVisibility(0)

        outer_actor = p.add_mesh(
            outer_recon_thick, scalars="thickness_recon", clim=THICK_CLIM, cmap=THICK_CMAP,
            opacity=OUTER_OPACITY, show_edges=False, show_scalar_bar=False,
            culling=False, name="OUTER"
        )
        recon_actor = p.add_mesh(
            recon_thick, scalars="thickness_recon", clim=THICK_CLIM, cmap=THICK_CMAP,
            opacity=RECON_OPACITY, show_edges=False, show_scalar_bar=True,
            scalar_bar_args={"title": "Thickness (mm)", "n_labels": 6},
            culling=False, name="RECON"
        )
        _current_pick_mesh = recon_thick
        _current_pick_scalar = "thickness_recon"
        picker.InitializePickList()
        picker.PickFromListOn()
        picker.AddPickList(recon_actor)

    # ---------------- View 3 ----------------
    elif state["view"] == 3:
        inner_actor = p.add_mesh(
            inner_eros, scalars="erosion", clim=EROSION_CLIM, cmap=EROSION_CMAP,
            show_edges=False, show_scalar_bar=True,
            scalar_bar_args={"title": "Erosion depth map (mm)", "n_labels": 6},
            culling=False, name="INNER"
        )
        outer_actor = p.add_mesh(
            outer_meas, color="white", opacity=0.10,
            show_edges=False, show_scalar_bar=False,
            culling=False, name="OUTER"
        )
        recon_actor = p.add_mesh(
            recon_thick, color="lightgray", opacity=RECON_OPACITY,
            show_edges=True, line_width=1, show_scalar_bar=False,
            culling=False, name="RECON"
        )
        _current_pick_mesh = inner_eros
        _current_pick_scalar = "erosion"
        picker.InitializePickList()
        picker.PickFromListOn()
        picker.AddPickList(inner_actor)

    # ---------------- View 4 (Zones) ----------------
    else:
        inner_actor = p.add_mesh(
            inner_eros, scalars="erosion", clim=EROSION_CLIM, cmap=EROSION_CMAP,
            show_edges=False, show_scalar_bar=True,
            scalar_bar_args={"title": "Erosion depth map (mm)", "n_labels": 6},
            culling=False, name="INNER"
        )
        outer_actor = p.add_mesh(
            outer_meas, color="white", opacity=0.08,
            show_edges=False, show_scalar_bar=False,
            culling=False, name="OUTER"
        )
        recon_actor = p.add_mesh(
            recon_thick, color="lightgray", opacity=0.25,
            show_edges=True, line_width=1, show_scalar_bar=False,
            culling=False, name="RECON"
        )

        # WHITE zone split lines on top of inferno
        if zone_lines is not None and zone_lines.n_points > 0:
            zone_lines_actor = p.add_mesh(
                zone_lines,
                color="white",
                line_width=5,
                name="ZONE_LINES",
                render_lines_as_tubes=True,
            )

        # Zone labels anchored at vertices (NOT floating)
        for i, (pos, txt) in enumerate(zone_labels):
            nm = f"ZLBL_{i:03d}"
            p.add_point_labels(
                np.array([pos]),
                [txt],
                font_size=25,
                point_size=1,
                shape_opacity=0.7,
                shape_color= "white",
                always_visible=True,
                text_color="black",
                name=nm,
                pickable=False,
                reset_camera=False,
            )
            zone_label_actors.append(nm)

        _current_pick_mesh = inner_eros
        _current_pick_scalar = "erosion"
        picker.InitializePickList()
        picker.PickFromListOn()
        picker.AddPickList(inner_actor)

    # box + bounds
    bbox_actor = p.add_mesh(two0.bounding_box(), style="wireframe", line_width=3, name="BBOX")
    bounds_actor = p.add_mesh(pv.Box(bounds=two0.bounds), style="wireframe", line_width=1, name="BOUNDS")

    vis = 1 if state["show_box"] else 0
    bbox_actor.SetVisibility(vis)
    bounds_actor.SetVisibility(vis)

    if state["show_box"]:
        try:
            p.show_axes()
        except Exception:
            pass
    else:
        try:
            p.hide_axes()
        except Exception:
            pass

    # HUD
    hud = (
        f"{dims_text}\n"
        "Views: 1=Measured thickness  2=Recon thickness (inner hidden)  3=Erosion  4=Erosion+Zones\n"
        "Right-click: sample | U undo | C clear\n"
        "B: box+axes | O: outer | F2: scalar bar layout\n"
        f"Recon offsets (mm): X={OFFSET_X}, Y={OFFSET_Y}, Z={OFFSET_Z}\n"
        f"Zone 1 bimodal? peaks(raw)={rep['kde_peaks_raw']} peaks(filt)={rep['kde_peaks_filtered']} "
        f"BC(filt)={rep['BC_filtered']}"
    )
    p.add_text(hud, position="upper_left", font_size=11, name="HUD")

    # scalar bar layout
    sb = get_scalar_bar_actor(p)
    if sb is not None:
        apply_scalarbar_layout(sb, state["scalar_vertical"])
        try:
            sb.SetTitleRatio(0.18)
        except Exception:
            pass

    # outer visibility toggle
    if outer_actor is not None:
        outer_actor.SetVisibility(1 if state["show_outer"] else 0)

    p.render()


def add_sample_label(world_pos: np.ndarray, value: float):
    base = f"sample_{len(_samples):05d}"
    p.add_point_labels(
        np.array([world_pos]),
        [f"{value:.3f}"],
        point_size=SAMPLE_LABEL_POINT_SIZE,
        font_size=SAMPLE_LABEL_FONT_SIZE,
        shape_opacity=SAMPLE_LABEL_SHAPE_OPACITY,
        always_visible=True,
        name=base,
        pickable=False,
        reset_camera=False,
    )
    _samples.append(base)


def on_right_click(obj, event):
    if _current_pick_mesh is None or _current_pick_scalar is None:
        return

    x, y = p.iren.get_event_position()
    picker.Pick(x, y, 0, p.renderer)

    cell_id = picker.GetCellId()
    if cell_id < 0:
        return

    pick_pos = np.array(picker.GetPickPosition(), dtype=float)
    mesh = _current_pick_mesh
    scalar = _current_pick_scalar

    pt_id = picker.GetPointId()
    if pt_id >= 0 and pt_id < mesh.n_points:
        val = float(mesh[scalar][pt_id])
    else:
        cell = mesh.extract_cells([cell_id])
        ids = cell.cell_point_ids(0)
        pts = mesh.points[np.array(ids)]
        j = int(np.argmin(np.linalg.norm(pts - pick_pos, axis=1)))
        pid = int(ids[j])
        val = float(mesh[scalar][pid])

    add_sample_label(pick_pos, val)
    p.render()


p.iren.add_observer("RightButtonPressEvent", on_right_click)


def remove_named_bundle(base: str):
    candidates = [base, f"{base}_points", f"{base}-points", f"{base}_labels", f"{base}-labels"]
    for name in candidates:
        try:
            p.remove_actor(name)
        except Exception:
            pass


def undo_last_sample():
    if not _samples:
        return
    base = _samples.pop()
    remove_named_bundle(base)
    p.render()


def clear_all_samples():
    while _samples:
        base = _samples.pop()
        remove_named_bundle(base)
    p.render()


def toggle_box():
    state["show_box"] = not state["show_box"]
    rebuild_scene()


def toggle_outer():
    state["show_outer"] = not state["show_outer"]
    if outer_actor is not None:
        outer_actor.SetVisibility(1 if state["show_outer"] else 0)
    p.render()


def toggle_scalarbar_layout():
    state["scalar_vertical"] = not state["scalar_vertical"]
    sb = get_scalar_bar_actor(p)
    if sb is not None:
        apply_scalarbar_layout(sb, state["scalar_vertical"])
        p.render()


def set_view(v: int):
    state["view"] = int(v)
    rebuild_scene()


def on_key_press_vtk(obj, event):
    k = obj.GetKeySym()

    if k == "1":
        set_view(1); return
    if k == "2":
        set_view(2); return
    if k == "3":
        set_view(3); return
    if k == "4":
        set_view(4); return

    if k in ("u", "U"):
        undo_last_sample(); return
    if k in ("c", "C"):
        clear_all_samples(); return

    if k in ("b", "B"):
        toggle_box(); return
    if k in ("o", "O"):
        toggle_outer(); return

    if k == "F2":
        toggle_scalarbar_layout(); return


iren = vtk_interactor(p)
iren.AddObserver("KeyPressEvent", on_key_press_vtk, 1.0)

rebuild_scene()
p.show()



# ============================================================
# POST: zone-wise stats + histograms (ALL zones) + Zone1 bimodality figure
# (runs after the PyVista window closes)
# ============================================================
erosion_depths = np.asarray(inner_eros["erosion"], dtype=float)
zones = np.asarray(inner_eros["zone_id"], dtype=int)

# ---- Controls ----

PLOT_ALL_ZONE_HISTS = True
SHOW_ZONE0_STATS = False   # Zone 0 is near-zero intrados erosion and is hidden by default
PLOT_ZONE0_HIST = False    # Usually a narrow spike near zero; enable only if specifically needed


MIN_POINTS_PER_ZONE = 80
BINS = 51
PLOT_P95 = True
MIN_PLOT_MAX = 2.5

SAVE_ZONE_HIST_FIGURES = True
SAVE_ZONE1_BIMODALITY_FIGURE = True





def plot_hist_with_best_fit(d, zone_id, noise_thr, title_prefix="Zone"):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < MIN_POINTS_PER_ZONE:
        return

    if PLOT_P95:
        p95 = float(np.percentile(d, 95))
        plot_max = max(MIN_PLOT_MAX, float(np.round(p95 + 0.5, 2)))
    else:
        plot_max = float(max(d.max(), MIN_PLOT_MAX))

    d_plot = d[d <= plot_max]
    bins = np.linspace(0, plot_max, BINS)

    fit = fit_candidates(d)
    best_name = None
    best_dist = None
    best_params = None
    best_ks = np.nan

    if fit is not None:
        best_name, best_ks, best_p, best_params, best_dist, _ = fit

    plt.figure(figsize=(10, 6))
    plt.hist(d_plot, bins=bins, density=True, alpha=0.55, edgecolor="black")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, plot_max)

    if best_dist is not None:
        x = np.linspace(0, plot_max, 400)
        pdf = best_dist.pdf(x, *best_params)
        plt.plot(x, pdf, linewidth=2, label=f"{best_name} fit (KS={best_ks:.3f})")
        plt.legend()

    # Optional: overlay Normal fit as a dashed reference
    try:
        mu_n, sigma_n = stats.norm.fit(d)
        x = np.linspace(0, plot_max, 400)
        pdf_n = stats.norm.pdf(x, mu_n, sigma_n)
        plt.plot(x, pdf_n, linestyle="--", linewidth=1.5, label="Normal reference")
        plt.legend()
    except Exception:
        pass

    plt.title(
        f"{title_prefix} {zone_id} erosion distribution (filtered ≥ {noise_thr:.2f} mm)\n"
        f"N={d.size}",
        fontsize=14
    )
    plt.xlabel("Erosion depth (mm)")
    plt.ylabel("Probability density")

    if SAVE_ZONE_HIST_FIGURES:
        out_path = FIGURES_DIR / f"zone_{zone_id}_histogram.png"
        plt.savefig(str(out_path), dpi=300, bbox_inches="tight")

    plt.show()
    
zone_stats_rows = []
print("\n=== Zone-wise erosion stats ===")
maxz = int(np.max(zones))

for z in range(0, maxz + 1):
    if z == 0 and not SHOW_ZONE0_STATS:
        continue

    d = erosion_depths[zones == z]
    d = d[np.isfinite(d)]

    if z != 0:
        d = d[d >= NOISE_THRESHOLD_MM]

    if d.size < 50:
        print(f"Zone {z}: n={d.size} (skip)")
        continue

    p95 = float(np.percentile(d, 95))
    mu = float(np.mean(d))
    med = float(np.median(d))
    mx = float(np.max(d))

    print(f"Zone {z}: n={d.size}  mean={mu:.3f}  med={med:.3f}  p95={p95:.3f}  max={mx:.3f}")

    best_name = ""
    ks = np.nan
    pval = np.nan

    fit = fit_candidates(d)
    if fit is not None:
        best_name, ks, pval, params, dist, _ = fit
        print(f"         best fit: {best_name}  KS={ks:.4f} p={pval:.4f}")
    else:
        print("         best fit: (fit failed)")

    norm_stat, norm_pval, norm_method = normality_report(d)
    print(
        f"         normality ({norm_method}): "
        f"stat={norm_stat:.4f} p={norm_pval:.4e}"
    )

    zone_stats_rows.append([
        z, d.size, mu, med, p95, mx,
        best_name, ks, pval,
        norm_method, norm_stat, norm_pval
    ])
    
    
       
zone_stats_csv = CSV_DIR / "zone_erosion_statistics.csv"

zone_stats_header = (
    "zone_id,n_points,mean_mm,median_mm,p95_mm,max_mm,"
    "best_fit,ks_stat,ks_pvalue,"
    "normality_method,normality_stat,normality_pvalue"
)

np.savetxt(
    str(zone_stats_csv),
    np.array(zone_stats_rows, dtype=object),
    fmt="%s",
    delimiter=",",
    header=zone_stats_header,
    comments=""
)
print(f"\nSaved zone statistics CSV: {zone_stats_csv}")    
    
    
# 2) Plot histogram+best-fit for ALL zones (this is what was missing)

if PLOT_ALL_ZONE_HISTS:
    for z in range(0, maxz + 1):
        if z == 0 and not PLOT_ZONE0_HIST:
            continue

        d = erosion_depths[zones == z]
        d = d[np.isfinite(d)]

        if z != 0:
            d = d[d >= NOISE_THRESHOLD_MM]

        plot_hist_with_best_fit(d, z, NOISE_THRESHOLD_MM, title_prefix="Zone")



# 3) Zone 1 bimodality figure

z1 = erosion_depths[zones == 1]
z1 = z1[np.isfinite(z1)]
z1f = z1[z1 >= NOISE_THRESHOLD_MM]

if z1f.size >= 50:
    peaks, xx, yy = kde_peak_count(z1f)
    bc = bimodality_coefficient(z1f)

    plot_max = max(2.5, float(np.percentile(z1f, 95)) + 0.5)
    plot_max = float(np.round(plot_max, 2))
    bins = np.linspace(0, plot_max, 51)

    plt.figure(figsize=(10, 6))
    plt.hist(z1f[z1f <= plot_max], bins=bins, density=True, alpha=0.55, edgecolor="black")
    plt.title(
        f"Zone 1 bimodality check (filtered ≥ {NOISE_THRESHOLD_MM:.2f} mm)\n"
        f"KDE peaks={peaks}, BC={bc:.3f}",
        fontsize=14,
    )
    plt.xlabel("Erosion depth (mm)")
    plt.ylabel("Probability density")
    plt.xlim(0, plot_max)
    plt.grid(True, alpha=0.3)

    if xx is not None:
        plt.plot(xx, yy, linewidth=2, label="KDE")
        plt.legend()

    if SAVE_ZONE1_BIMODALITY_FIGURE:
        out_path = FIGURES_DIR / "zone_1_bimodality.png"
        plt.savefig(str(out_path), dpi=300, bbox_inches="tight")

    plt.show()

else:
    print("\nZone 1 has too few points above noise threshold for bimodality plotting.")

zone1_csv = CSV_DIR / "zone1_bimodality_report.csv"
with open(zone1_csv, "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    for k, v in rep.items():
        f.write(f"{k},{v}\n")
print(f"Saved Zone 1 bimodality CSV: {zone1_csv}")


# ============================================================
# Save erosion depth data to CSV
# ============================================================

# Full erosion data for all points
all_pts = np.asarray(inner_eros.points, dtype=float)
all_erosion = np.asarray(inner_eros["erosion"], dtype=float)
all_zones = np.asarray(inner_eros["zone_id"], dtype=int)

all_data = np.c_[all_pts, all_erosion, all_zones]
all_csv = CSV_DIR / "erosion_depth_all_points.csv"
all_header = "x_mm,y_mm,z_mm,erosion_mm,zone_id"

np.savetxt(
    str(all_csv),
    all_data,
    delimiter=",",
    header=all_header,
    comments=""
)
print(f"Saved full erosion depth CSV: {all_csv}")


# ============================================================
# Save erosion depth data to CSV
# ============================================================

all_pts = np.asarray(inner_eros.points, dtype=float)
all_erosion = np.asarray(inner_eros["erosion"], dtype=float)
all_zones = np.asarray(inner_eros["zone_id"], dtype=int)

all_data = np.c_[all_pts, all_erosion, all_zones]
all_csv = CSV_DIR / "erosion_depth_all_points.csv"
all_header = "x_mm,y_mm,z_mm,erosion_mm,zone_id"

np.savetxt(
    str(all_csv),
    all_data,
    delimiter=",",
    header=all_header,
    comments=""
)
print(f"Saved full erosion depth CSV: {all_csv}")

# Per-zone raw data
for z in np.unique(all_zones):
    zone_mask = (all_zones == z)
    zone_data = np.c_[all_pts[zone_mask], all_erosion[zone_mask], all_zones[zone_mask]]

    zone_csv = CSV_DIR / f"erosion_depth_zone_{z}.csv"
    np.savetxt(
        str(zone_csv),
        zone_data,
        delimiter=",",
        header=all_header,
        comments=""
    )
    print(f"Saved zone {z} erosion depth CSV: {zone_csv}")

# Per-zone filtered data
for z in np.unique(all_zones):
    if z == 0:
        continue

    zone_mask = (all_zones == z) & (all_erosion >= NOISE_THRESHOLD_MM)
    if np.sum(zone_mask) == 0:
        continue

    zone_data = np.c_[all_pts[zone_mask], all_erosion[zone_mask], all_zones[zone_mask]]

    zone_csv = CSV_DIR / f"erosion_depth_zone_{z}_filtered.csv"
    np.savetxt(
        str(zone_csv),
        zone_data,
        delimiter=",",
        header=all_header,
        comments=""
    )
    print(f"Saved filtered zone {z} erosion depth CSV: {zone_csv}")