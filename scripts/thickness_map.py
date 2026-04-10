import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree as KDTree
from pathlib import Path


# ----------------------------
# Helpers
# ----------------------------
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


def compute_thickness_nn(inner: pv.PolyData, outer: pv.PolyData):
    outer_tree = KDTree(outer.points)
    d_inner, _ = outer_tree.query(inner.points, k=1)

    inner_out = inner.copy()
    inner_out["thickness"] = d_inner.astype(np.float64)

    inner_tree = KDTree(inner.points)
    _, idx_inner = inner_tree.query(outer.points, k=1)

    outer_out = outer.copy()
    outer_out["thickness"] = inner_out["thickness"][idx_inner]
    return inner_out, outer_out


def box_dims_from_bounds(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (xmax - xmin), (ymax - ymin), (zmax - zmin)


def pca_frame(points: np.ndarray):
    C = points.mean(axis=0)
    X = points - C
    cov = (X.T @ X) / max(1, (len(points) - 1))
    w, V = np.linalg.eigh(cov)
    V = V[:, np.argsort(w)[::-1]]
    e1, e2, e3 = V[:, 0], V[:, 1], V[:, 2]
    if np.dot(np.cross(e1, e2), e3) < 0:
        e3 = -e3
    return C, e1, e2, e3


def remove_named_bundle(plotter: pv.Plotter, base: str):
    candidates = [base, f"{base}_points", f"{base}-points", f"{base}_labels", f"{base}-labels"]
    for name in candidates:
        try:
            plotter.remove_actor(name)
        except Exception:
            pass


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
    """
    Find the first vtkScalarBarActor in the renderer.
    This bar is created by add_mesh(show_scalar_bar=True) and shares the same LUT/clim.
    """
    actors2d = plotter.renderer.GetActors2D()
    actors2d.InitTraversal()
    for _ in range(actors2d.GetNumberOfItems()):
        a = actors2d.GetNextActor2D()
        if isinstance(a, vtk.vtkScalarBarActor):
            return a
    return None


def apply_scalarbar_layout(sb: vtk.vtkScalarBarActor, vertical: bool):
    """
    Set scalar bar placement/orientation WITHOUT changing LUT or range.
    Coordinates are normalized viewport (0..1).
    """
    pos = sb.GetPositionCoordinate()
    pos.SetCoordinateSystemToNormalizedViewport()

    if vertical:
        # Right side
        sb.SetOrientationToVertical()
        pos.SetValue(0.88, 0.12)
        sb.SetWidth(0.10)
        sb.SetHeight(0.78)
    else:
        # Bottom
        sb.SetOrientationToHorizontal()
        pos.SetValue(0.20, 0.05)
        sb.SetWidth(0.60)
        sb.SetHeight(0.08)


# ----------------------------
# Settings
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_STL = BASE_DIR / "data" /  "cut_initial_thickness_1_7_mm.stl"

THICK_CLIM = (0.0, 13.0)   # <-- bar range
OUTER_OPACITY = 1.0

GRID_SPACING_MM_DEFAULT = 30
GRID_SPACING_MIN = 2.0
GRID_SPACING_MAX = 50.0

GRID_POINT_SIZE = 10
GRID_ACTOR_NAME = "UT_GRID_POINTS"
GRID_CSV_NAME = BASE_DIR / "results" / "csv" / "ut_grid_thickness.csv"

GRID_LABEL_POINT_SIZE = 8
GRID_LABEL_FONT_SIZE = 15
GRID_LABEL_SHAPE_OPACITY = 0.25

# Manual point sampling label appearance
SAMPLE_LABEL_FONT_SIZE = 18     # ← same idea as GRID_LABEL_FONT_SIZE
SAMPLE_LABEL_POINT_SIZE = 10
SAMPLE_LABEL_SHAPE_OPACITY = 0.35


# ----------------------------
# Load + shift to origin
# ----------------------------

if not INPUT_STL.exists():
    raise FileNotFoundError(f"Input STL not found: {INPUT_STL}")

mesh = ensure_polydata(pv.read(str(INPUT_STL)))



shift = np.array(mesh.center, dtype=float)
mesh0 = mesh.copy()
mesh0.points = mesh0.points - shift

s0, s1 = split_two_surfaces(mesh0)
inner, outer = pick_inner_outer_by_radius(s0, s1)

inner_t, outer_t = compute_thickness_nn(inner, outer)

bbox = mesh0.bounding_box()
dx, dy, dz = box_dims_from_bounds(mesh0.bounds)
dims_text = f"Box ΔX={dx:.2f} mm  ΔY={dy:.2f} mm  ΔZ={dz:.2f} mm"


# ----------------------------
# Plot
# ----------------------------
p = pv.Plotter()
_scalar_vertical = True

# Inner mesh OWNS the scalar bar + LUT + clim
inner_actor = p.add_mesh(
    inner_t,
    scalars="thickness",
    clim=THICK_CLIM,
    show_edges=False,
    show_scalar_bar=True,
    scalar_bar_args={"title": "Thickness (mm)", "n_labels": 6},
    culling=False,
    name="inner",
)

# Outer overlay
outer_actor = p.add_mesh(
    outer_t,
    scalars="thickness",
    clim=THICK_CLIM,
    opacity=OUTER_OPACITY,
    show_edges=False,
    show_scalar_bar=False,
    culling=False,
    name="outer",
)

# Box + axes
bbox_actor = p.add_mesh(bbox, style="wireframe", line_width=3, name="bbox")
bounds_actor = p.add_mesh(pv.Box(bounds=mesh0.bounds), style="wireframe", line_width=1, name="bounds")
p.add_axes()
dims_actor = p.add_text(dims_text, position="upper_right", font_size=11)

p.add_text(
    "Right-click: sample (inner)\n"
    "U: undo last sample   C: clear samples\n"
    "B: toggle box+axes frame\n"
    "O: toggle outer overlay\n"
    "G: toggle UT grid (outer) + numbers\n"
    "[ or \\ : finer grid   ] or / : coarser grid\n"
    "K: export grid CSV\n"
    "F2: toggle scalar bar vertical/horizontal",
    position="upper_left",
    font_size=11,
)

# Grab the scalar bar actor created by inner_actor and set initial layout
scalar_bar_actor = get_scalar_bar_actor(p)
if scalar_bar_actor is not None:
    apply_scalarbar_layout(scalar_bar_actor, _scalar_vertical)

 # ---- TITLE OFFSET TUNING ----
    scalar_bar_actor.SetTitleRatio(1)
    try:
        scalar_bar_actor.SetVerticalTitleSeparation(10)
    except AttributeError:
        pass

    title_prop = scalar_bar_actor.GetTitleTextProperty()
    title_prop.SetFontSize(16)


# ----------------------------
# Right-click sampling on INNER
# ----------------------------
picker = vtk.vtkCellPicker()
picker.SetTolerance(0.0005)
picker.PickFromListOn()
picker.AddPickList(inner_actor)

_samples = []

def on_right_click(obj, event):
    x, y = p.iren.get_event_position()
    picker.Pick(x, y, 0, p.renderer)
    cell_id = picker.GetCellId()
    if cell_id < 0:
        return
    pick_pos = np.array(picker.GetPickPosition(), dtype=float)

    pt_id = picker.GetPointId()
    if pt_id >= 0:
        val = float(inner_t["thickness"][pt_id])
    else:
        cell = inner_t.extract_cells([cell_id])
        ids = cell.cell_point_ids(0)
        pts = inner_t.points[np.array(ids)]
        j = int(np.argmin(np.linalg.norm(pts - pick_pos, axis=1)))
        pid = int(ids[j])
        val = float(inner_t["thickness"][pid])

    base = f"sample_{len(_samples):05d}"
    p.add_point_labels(
        np.array([pick_pos]),
        [f"{val:.3f}"],
        point_size= SAMPLE_LABEL_POINT_SIZE,
        font_size= SAMPLE_LABEL_FONT_SIZE,
        shape_opacity= SAMPLE_LABEL_SHAPE_OPACITY,
        always_visible=True,
        name=base,
        pickable=False,
        reset_camera=False,
    )
    
   
    
    _samples.append(base)
    p.render()

p.iren.add_observer("RightButtonPressEvent", on_right_click)

def undo_last_sample():
    if not _samples:
        return
    base = _samples.pop()
    remove_named_bundle(p, base)
    p.render()

def clear_all_samples():
    while _samples:
        base = _samples.pop()
        remove_named_bundle(p, base)
    p.render()


# ----------------------------
# UT grid on OUTER
# ----------------------------
_state = {"box": True, "outer": True, "grid": False}
_grid_spacing = float(GRID_SPACING_MM_DEFAULT)

_ut_grid_cache = {"points": None, "thickness": None, "uv": None}
_ut_label_names = []

def build_ut_grid_on_outer(spacing_mm: float):
    P = outer_t.points
    C, e1, e2, _ = pca_frame(P)
    U = (P - C) @ e1
    V = (P - C) @ e2
    UV = np.c_[U, V]
    uv_tree = KDTree(UV)

    u_nodes = np.arange(U.min(), U.max() + 0.5 * spacing_mm, spacing_mm)
    v_nodes = np.arange(V.min(), V.max() + 0.5 * spacing_mm, spacing_mm)
    UU, VV = np.meshgrid(u_nodes, v_nodes, indexing="xy")
    nodes_uv = np.c_[UU.ravel(), VV.ravel()]

    _, idx = uv_tree.query(nodes_uv, k=1)
    idx_unique = np.unique(np.asarray(idx, dtype=int))

    pts = P[idx_unique]
    thick = outer_t["thickness"][idx_unique].astype(float)
    uv_unique = UV[idx_unique]
    return pts, thick, uv_unique

def clear_ut_grid_labels():
    global _ut_label_names
    for base in _ut_label_names:
        remove_named_bundle(p, base)
    _ut_label_names = []

def show_ut_grid():
    global _ut_grid_cache, _ut_label_names
    pts, thick, uv = build_ut_grid_on_outer(_grid_spacing)
    _ut_grid_cache["points"] = pts
    _ut_grid_cache["thickness"] = thick
    _ut_grid_cache["uv"] = uv

    try:
        p.remove_actor(GRID_ACTOR_NAME)
    except Exception:
        pass
    clear_ut_grid_labels()

    grid_pd = pv.PolyData(pts)
    grid_pd["thickness"] = thick
    p.add_mesh(
        grid_pd,
        render_points_as_spheres=True,
        point_size=GRID_POINT_SIZE,
        scalars="thickness",
        clim=THICK_CLIM,
        show_scalar_bar=False,
        name=GRID_ACTOR_NAME,
        pickable=False,
        opacity=1.0,
    )

    for k in range(len(pts)):
        base = f"ut_{k:06d}"
        p.add_point_labels(
            np.array([pts[k]]),
            [f"{float(thick[k]):.2f}"],
            point_size=GRID_LABEL_POINT_SIZE,
            font_size=GRID_LABEL_FONT_SIZE,
            shape_opacity=GRID_LABEL_SHAPE_OPACITY,
            always_visible=True,
            name=base,
            pickable=False,
            reset_camera=False,
        )
        _ut_label_names.append(base)
    p.render()

def hide_ut_grid():
    try:
        p.remove_actor(GRID_ACTOR_NAME)
    except Exception:
        pass
    clear_ut_grid_labels()
    p.render()

def toggle_ut_grid():
    _state["grid"] = not _state["grid"]
    if _state["grid"]:
        show_ut_grid()
    else:
        hide_ut_grid()

def grid_spacing_down():
    global _grid_spacing
    _grid_spacing = max(GRID_SPACING_MIN, _grid_spacing - 1.0)
    if _state["grid"]:
        show_ut_grid()

def grid_spacing_up():
    global _grid_spacing
    _grid_spacing = min(GRID_SPACING_MAX, _grid_spacing + 1.0)
    if _state["grid"]:
        show_ut_grid()

def export_ut_grid_csv():
    pts = _ut_grid_cache["points"]
    thick = _ut_grid_cache["thickness"]
    uv = _ut_grid_cache["uv"]
    if pts is None or thick is None:
        pts, thick, uv = build_ut_grid_on_outer(_grid_spacing)
    data = np.c_[pts, thick, uv]
    header = "x_mm,y_mm,z_mm,thickness_mm,u_pca_mm,v_pca_mm"
    GRID_CSV_NAME.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(GRID_CSV_NAME), data, delimiter=",", header=header, comments="")
    
    print(f"[UT GRID] Saved {GRID_CSV_NAME} ({len(pts)} knots, spacing={_grid_spacing:.1f} mm).")

# ----------------------------
# Box/outer toggles
# ----------------------------
def _try_hide_axes():
    try:
        p.hide_axes()
    except Exception:
        pass

def _try_show_axes():
    try:
        p.show_axes()
    except Exception:
        pass

def toggle_box():
    _state["box"] = not _state["box"]
    vis = 1 if _state["box"] else 0
    bbox_actor.SetVisibility(vis)
    bounds_actor.SetVisibility(vis)
    try:
        dims_actor.SetVisibility(vis)
    except Exception:
        pass
    if _state["box"]:
        _try_show_axes()
    else:
        _try_hide_axes()
    p.render()

def toggle_outer():
    _state["outer"] = not _state["outer"]
    outer_actor.SetVisibility(1 if _state["outer"] else 0)
    p.render()

# ----------------------------
# Scalar bar toggle (F2) — SAME LUT, SAME RANGE
# ----------------------------
def toggle_scalarbar_orientation():
    global _scalar_vertical, scalar_bar_actor
    _scalar_vertical = not _scalar_vertical
    if scalar_bar_actor is None:
        scalar_bar_actor = get_scalar_bar_actor(p)
    if scalar_bar_actor is not None:
        apply_scalarbar_layout(scalar_bar_actor, _scalar_vertical)
        p.render()

# ----------------------------
# KeyPress handler
# ----------------------------
def on_key_press_vtk(obj, event):
    k = obj.GetKeySym()

    if k == "F2":
        toggle_scalarbar_orientation()
        return

    if k in ("g", "G"):
        toggle_ut_grid()
        return
    if k in ("k", "K"):
        export_ut_grid_csv()
        return
    if k in ("bracketleft", "backslash"):
        grid_spacing_down()
        return
    if k in ("bracketright", "slash"):
        grid_spacing_up()
        return
    if k in ("u", "U"):
        undo_last_sample()
        return
    if k in ("c", "C"):
        clear_all_samples()
        return
    if k in ("b", "B"):
        toggle_box()
        return
    if k in ("o", "O"):
        toggle_outer()
        return

iren = vtk_interactor(p)
iren.AddObserver("KeyPressEvent", on_key_press_vtk, 1.0)

p.show()
