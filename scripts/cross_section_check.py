"""
cross_section_check.py

Cross-section verification for two aligned STL surfaces stored in a single file.
The script:
1. separates the two largest connected components,
2. identifies inner and outer surfaces,
3. estimates the bend path using midpoint binning and PCA,
4. creates cross-sections normal to the estimated bend path,
5. visualizes reconstructed and eroded section curves.

Reconstructed sections are shown in red.
Eroded sections are shown in blue.

"""


import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree as KDTree
from pathlib import Path


# ============================================================
#                          SETTINGS
# ============================================================

# input, aligned in Blender and exported in STL format. 

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_STL = BASE_DIR / "data" / "New_reconstructed_aligned_1.stl" 

# How many cross-sections you want to visualize (planes normal to bend path)
N_STATIONS = 15

# Path estimation (more bins = finer angular sampling; too many bins needs lower MIN_BIN_POINTS)
N_THETA_BINS = 120
MIN_BIN_POINTS = 120  # lower if you trimmed heavily or have sparse areas

# IMPORTANT: elbows are NOT 360° sweep. Restrict the bend window.
# Typical good starting points:
#   90° elbow:  THETA_START_DEG=-60,  THETA_END_DEG=+60
#  180° elbow:  THETA_START_DEG=-120, THETA_END_DEG=+120
USE_THETA_WINDOW = True
THETA_START_DEG = 0
THETA_END_DEG   =  360

# Reject bad bins whose radius deviates a lot (keeps path on the bend)
RADIAL_OUTLIER_SIGMA = 100  # 2.0–3.5 typical
SMOOTH_ITERS = 20         # 5–15 typical

# Display controls
SHOW_SURFACES = True
INNER_OPACITY = 0.12
OUTER_OPACITY = 0.2
INNER_LINE_WIDTH = 4
OUTER_LINE_WIDTH = 4

SHOW_TANGENT_ARROWS = False  # set True to debug planes
ARROW_SCALE_FRAC = 0.12      # fraction of model length


# ============================================================
#                       Helpers
# ============================================================
def ensure_polydata(mesh):
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.clean()


def split_two_largest_components(mesh: pv.PolyData):
    conn = mesh.connectivity()
    ids = np.asarray(conn["RegionId"])
    u, c = np.unique(ids, return_counts=True)
    order = np.argsort(c)[::-1]
    if len(order) < 2:
        raise RuntimeError("STL must contain at least two connected components (inner + outer).")
    reg0, reg1 = u[order[0]], u[order[1]]
    s0 = conn.threshold([reg0, reg0], scalars="RegionId") \
         .extract_surface(algorithm="dataset_surface").clean()

    s1 = conn.threshold([reg1, reg1], scalars="RegionId") \
         .extract_surface(algorithm="dataset_surface").clean()
    return s0, s1


def pick_inner_outer_by_radius(s0: pv.PolyData, s1: pv.PolyData):
    c = np.vstack([s0.points, s1.points]).mean(axis=0)
    r0 = np.mean(np.linalg.norm(s0.points - c, axis=1))
    r1 = np.mean(np.linalg.norm(s1.points - c, axis=1))
    return (s1, s0) if r0 > r1 else (s0, s1)


def pca_frame(points: np.ndarray):
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(len(points) - 1, 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1.0
    return c, V  # columns are principal axes


def compute_tangents(P: np.ndarray):
    """Discrete unit tangents using central differences."""
    T = np.zeros_like(P)
    n = len(P)
    for i in range(n):
        if i == 0:
            d = P[1] - P[0]
        elif i == n - 1:
            d = P[-1] - P[-2]
        else:
            d = P[i+1] - P[i-1]
        norm = np.linalg.norm(d)
        T[i] = d / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    return T


def resample_polyline(points: np.ndarray, n: int):
    """Resample polyline points by arc length."""
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 1e-12:
        return np.repeat(points[:1], n, axis=0)
    s_new = np.linspace(0.0, L, n)

    out = np.zeros((n, 3), dtype=float)
    for k, sk in enumerate(s_new):
        j = np.searchsorted(s, sk) - 1
        j = max(0, min(j, len(points) - 2))
        t = (sk - s[j]) / max(s[j+1] - s[j], 1e-12)
        out[k] = (1 - t) * points[j] + t * points[j+1]
    return out


def largest_polyline(slice_poly: pv.PolyData):
    """Keep only the largest connected polyline component from a slice."""
    if slice_poly is None or slice_poly.n_points == 0:
        return None
    conn = slice_poly.connectivity()
    if "RegionId" not in conn.point_data:
        return slice_poly
    ids = np.asarray(conn["RegionId"])
    u, c = np.unique(ids, return_counts=True)
    reg = u[np.argmax(c)]
    out = conn.threshold([reg, reg], scalars="RegionId") \
         .extract_surface(algorithm="dataset_surface").clean()
    return out if out.n_points > 0 else None


# ============================================================
# Load STL, split eroded/reconstructed
# ============================================================
mesh = ensure_polydata(pv.read(INPUT_STL))
s0, s1 = split_two_largest_components(mesh)
inner, outer = pick_inner_outer_by_radius(s0, s1)

all_pts = np.vstack([inner.points, outer.points])
centroid, R = pca_frame(all_pts)
e0, e1, e2 = R[:, 0], R[:, 1], R[:, 2]  # bend plane approx = span(e0,e1)

# ============================================================
# Midpoints (outer -> nearest inner)
# ============================================================
inner_tree = KDTree(inner.points)
_, idx_i = inner_tree.query(outer.points, k=1)
mid_pts = 0.5 * (outer.points + inner.points[idx_i])

# Project midpoints to bend plane coordinates (theta for binning)
v = mid_pts - centroid
u0 = v @ e0
u1 = v @ e1
theta = np.arctan2(u1, u0)  # [-pi, pi]


# ============================================================
# Optional: restrict theta sweep (strongly recommended)
# ============================================================
if USE_THETA_WINDOW:
    a0 = np.deg2rad(THETA_START_DEG)
    a1 = np.deg2rad(THETA_END_DEG)

    if a0 <= a1:
        mask = (theta >= a0) & (theta <= a1)
    else:
        # wrap-around interval
        mask = (theta >= a0) | (theta <= a1)

    mid_pts = mid_pts[mask]
    theta = theta[mask]

    if mid_pts.shape[0] < 1000:
        raise RuntimeError(
            "Not enough midpoints after theta window filtering.\n"
            "Try widening THETA_START/END or set USE_THETA_WINDOW=False."
        )


# ============================================================
# Estimate bend path by theta binning (robust)
# ============================================================
edges = np.linspace(-np.pi, np.pi, N_THETA_BINS + 1)

path_pts = []
path_theta = []

for k in range(N_THETA_BINS):
    m = (theta >= edges[k]) & (theta < edges[k+1])
    if np.count_nonzero(m) < MIN_BIN_POINTS:
        continue

    cbin = mid_pts[m].mean(axis=0)
    vbin = cbin - centroid
    th = np.arctan2(vbin @ e1, vbin @ e0)

    path_pts.append(cbin)
    path_theta.append(th)

path_pts = np.asarray(path_pts, dtype=float)
path_theta = np.asarray(path_theta, dtype=float)

if len(path_pts) < 8:
    raise RuntimeError(
        f"Could not estimate bend path reliably: only {len(path_pts)} bins passed MIN_BIN_POINTS.\n"
        f"Try: decrease MIN_BIN_POINTS, increase N_THETA_BINS, or widen theta window."
    )

# Sort by theta and unwrap to avoid +/-pi jump
order = np.argsort(path_theta)
path_pts = path_pts[order]
path_theta = path_theta[order]
path_theta_u = np.unwrap(path_theta)

# Outlier rejection by radius in bend plane (robust)
vv = path_pts - centroid
r = np.sqrt((vv @ e0) ** 2 + (vv @ e1) ** 2)

r_med = float(np.median(r))
r_mad = float(np.median(np.abs(r - r_med)) + 1e-12)
sigma = 1.4826 * r_mad  # MAD->sigma

keep = np.abs(r - r_med) <= RADIAL_OUTLIER_SIGMA * sigma
path_pts = path_pts[keep]
path_theta_u = path_theta_u[keep]

if len(path_pts) < 8:
    raise RuntimeError(
        f"Too many outliers removed: kept {len(path_pts)} bins.\n"
        f"Try: increase RADIAL_OUTLIER_SIGMA or decrease MIN_BIN_POINTS."
    )

# Sort again by unwrapped theta
order = np.argsort(path_theta_u)
path_pts = path_pts[order]

# Smooth the path to stabilize tangents
P = path_pts.copy()
for _ in range(int(SMOOTH_ITERS)):
    P2 = P.copy()
    P2[1:-1] = 0.25 * P[:-2] + 0.50 * P[1:-1] + 0.25 * P[2:]
    P = P2
path_pts = P

# Resample to stations and compute tangents
stations = resample_polyline(path_pts, int(N_STATIONS))
tangents = compute_tangents(stations)

# ---- Optional: extend stations to cover full model length (straight ends) ----
EXTEND_TO_BOUNDS = True
EXTEND_FRACTION = 0.09  # extend by ~8% of bounding-box diagonal on each side

if EXTEND_TO_BOUNDS:
    bounds = mesh.bounds
    diag = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
    ext = EXTEND_FRACTION * diag

    # extend first station backward along its tangent
    s0 = stations[0] - ext * tangents[0]
    # extend last station forward along its tangent
    s1 = stations[-1] + ext * tangents[-1]

    stations = np.vstack([s0, stations, s1])
    tangents = compute_tangents(stations)  # recompute tangents after extension

# ============================================================
# Plot slices with planes normal to bend path
# ============================================================
p = pv.Plotter()
p.set_background("white")

if SHOW_SURFACES:
    p.add_mesh(inner, color="white", opacity=INNER_OPACITY, show_edges=False)
    p.add_mesh(outer, color="white", opacity=OUTER_OPACITY, show_edges=False)

# Show estimated bend path curve
path_line = pv.Spline(stations, 250)
p.add_mesh(path_line, color="black", line_width=3, opacity = 0)
p.add_mesh(pv.PolyData(stations), color="black", point_size=10, opacity = 0, render_points_as_spheres=True)

# Debug: tangent arrows
if SHOW_TANGENT_ARROWS:
    arrow_scale = float(ARROW_SCALE_FRAC) * mesh.length
    for o, t in zip(stations, tangents):
        p.add_mesh(pv.Arrow(start=o, direction=t, scale=arrow_scale), color="green")

good = 0
for origin, n in zip(stations, tangents):
    # plane normal = tangent => plane is normal to bend path
    sli = largest_polyline(inner.slice(normal=n, origin=origin))
    slo = largest_polyline(outer.slice(normal=n, origin=origin))

    if sli is not None:
        p.add_mesh(sli, color="red", line_width=INNER_LINE_WIDTH)
    if slo is not None:
        p.add_mesh(slo, color="blue", line_width=OUTER_LINE_WIDTH)

    if (sli is not None) or (slo is not None):
        good += 1

p.add_text(
    "Cross-sections normal to estimated bend path\n"
    f"Stations: {N_STATIONS}, slices shown: {good}\n"
    f"Bins kept: {len(path_pts)} / {N_THETA_BINS}\n"
    f"Theta window: {USE_THETA_WINDOW}  ({THETA_START_DEG:.0f}° .. {THETA_END_DEG:.0f}°)\n"
    "Red=Reconstructed  Blue=Eroded  Black=estimated bend path (hidden by default)",
    position="upper_left",
    font_size=11,
    color="black",
)

p.show_axes()
p.reset_camera()
p.show()
