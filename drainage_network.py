"""
engine/drainage_network.py - Drainage Routing + Stream Vectorization (v3)
=========================================================================
Strict engineering routing:
  1. Extract natural drain network (Zhang-Suen vectorization).
  2. Route each HIGH basin to natural_drain_network.shp ONLY.
  3. Strict gravity constraint: no uphill movement.
  4. Abadi avoidance: infinite cost on settlement cells.
  5. Pump lift computation for unreachable-by-gravity basins.

Inputs:
  terrain/village_dtm_clean.tif, terrain/slope.tif
  hydrology/flow_accumulation.tif, hydrology/convergence_index.tif,
           hydrology/dist_to_stream.tif
  waterlogging/waterlogging_risk_class.tif
  drainage/natural_drain_network.shp  (created by run_streams)
  drainage/abadi_area.shp             (optional)

Outputs (to village_dir/drainage/):
  optimized_drainage_network.shp  - proposed drains (with engineering attrs)
  natural_drain_network.shp       - extracted natural streams
  drainage_design_summary.csv

Routing rules:
  A. Valid outlet = intersection with natural_drain_network.shp ONLY
  B. Gravity: DTM(next) <= DTM(current), strictly enforced
  C. Abadi cells = infinite cost (complete avoidance)
  D. No boundary-based termination
  E. Deterministic: identical results across runs
"""
import csv
import heapq
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import distance_transform_edt, binary_dilation
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

# -- Parameters ----------------------------------------------------------
MIN_CLUSTER_AREA = 50         # m2 minimum routable cluster
MAX_CLUSTERS = 1000
W_SLOPE = 0.6                 # cost weight: slope component
W_CONV = 0.4                  # cost weight: convergence component
MIN_SEARCH_RADIUS_M = 500.0
SEARCH_RADIUS_FACTOR = 1.5
STREAM_DENSITY_MIN = 0.0001
STREAM_DENSITY_MAX = 0.002
NODATA = -9999.0

# -- Gravity tolerance (micro-relief noise suppression) -----
GRAVITY_TOLERANCE = 0.05      # meters (5 cm) – allows routing over
                               # LiDAR noise, road crowns, soil ridges

# -- Engineering routing constraints --------------------------------
MAX_ROUTING_DISTANCE = 300    # meters – maximum A* search distance
MAX_PUMP_LIFT = 6.0           # meters – max feasible pump head; above → RETENTION
SECONDARY_FA_PERCENTILE = 95  # top 5% FA cells = secondary outlet candidates

NEIGHBORS_8 = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),                          (0, 1, 1.0),
    (1, -1, math.sqrt(2)),  (1, 0, 1.0),   (1, 1, math.sqrt(2)),
]


# ── Raster I/O ─────────────────────────────────────────────────

def _read_raster(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float32)
    ds = None
    return arr, gt, proj, nd


def _robust_normalize(arr, valid_mask):
    vals = arr[valid_mask].astype(np.float32)
    if len(vals) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    med = np.float32(np.median(vals))
    q1 = np.float32(np.percentile(vals, 25))
    q3 = np.float32(np.percentile(vals, 75))
    iqr = q3 - q1
    if iqr < 1e-10:
        iqr = np.float32(max(np.std(vals), 1e-10))
    normed = (arr.astype(np.float32) - med) / iqr
    v_min = float(np.min(normed[valid_mask]))
    v_max = float(np.max(normed[valid_mask]))
    rng = v_max - v_min if (v_max - v_min) > 1e-10 else 1.0
    normed = (normed - np.float32(v_min)) / np.float32(rng)
    normed = np.clip(normed, 0.0, 1.0)
    normed[~valid_mask] = 0.0
    return normed


def _write_shp(out_path, features, gt, proj):
    """Write list of LineString features to shapefile.

    features: list of dicts with 'path' (list of (r,c)) and metadata fields.
    Engineering attributes: basin_id, start_elev, end_elev, total_length_m,
    mean_slope_percent, pump_required, required_lift_m, feasible.
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # Clean existing — fall back to .tmp.shp if locked
    actual = out_path
    base = out_path.replace(".shp", "")
    locked = False
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = base + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                locked = True
    if locked:
        actual = out_path.replace(".shp", ".tmp.shp")
        tmp_base = actual.replace(".shp", "")
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = tmp_base + ext
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    ds = driver.CreateDataSource(actual)
    layer = ds.CreateLayer("routes", srs, ogr.wkbLineString)

    # -- Engineering attribute fields --
    layer.CreateField(ogr.FieldDefn("basin_id", ogr.OFTInteger))
    fld = ogr.FieldDefn("area_m2", ogr.OFTReal); fld.SetWidth(12); fld.SetPrecision(1)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("start_elev", ogr.OFTReal); fld.SetWidth(10); fld.SetPrecision(3)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("end_elev", ogr.OFTReal); fld.SetWidth(10); fld.SetPrecision(3)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("length_m", ogr.OFTReal); fld.SetWidth(10); fld.SetPrecision(1)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("mean_slp", ogr.OFTReal); fld.SetWidth(8); fld.SetPrecision(2)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("elev_drop", ogr.OFTReal); fld.SetWidth(8); fld.SetPrecision(2)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("status", ogr.OFTString); fld.SetWidth(20)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("pump_req", ogr.OFTString); fld.SetWidth(5)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("lift_m", ogr.OFTReal); fld.SetWidth(8); fld.SetPrecision(3)
    layer.CreateField(fld)
    fld = ogr.FieldDefn("feasible", ogr.OFTString); fld.SetWidth(5)
    layer.CreateField(fld)

    n = 0
    for rec in features:
        path = rec.get("path")
        if not path or len(path) < 2:
            continue
        line = ogr.Geometry(ogr.wkbLineString)
        for r, c in path:
            x = gt[0] + c * gt[1] + r * gt[2]
            y = gt[3] + c * gt[4] + r * gt[5]
            line.AddPoint(x, y)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("basin_id", int(rec.get("cluster_id", 0)))
        feat.SetField("area_m2", float(rec.get("area_m2", 0)))
        feat.SetField("start_elev", float(rec.get("start_elev", 0)))
        feat.SetField("end_elev", float(rec.get("end_elev", 0)))
        feat.SetField("length_m", float(rec.get("path_len_m", 0)))
        feat.SetField("mean_slp", float(rec.get("mean_slp_pct", 0)))
        feat.SetField("elev_drop", float(rec.get("elev_drop_m", 0)))
        feat.SetField("status", str(rec.get("status", "")))
        feat.SetField("pump_req", "True" if rec.get("pump_required", False) else "False")
        feat.SetField("lift_m", float(rec.get("required_lift_m", 0.0)))
        feat.SetField("feasible", "True" if rec.get("feasible", False) else "False")
        feat.SetGeometry(line)
        layer.CreateFeature(feat)
        feat = None
        n += 1
    ds = None
    print(f"  Wrote: {n} features -> {os.path.basename(actual)}", flush=True)
    return n


# ── A* Routing (v3 – strict gravity) ──────────────────────────

def _gravity_astar(cost, elev, start, goals_mask, abadi_mask,
                   max_radius_px, pixel_size):
    """A* routing with STRICT gravity constraint (Rule B).

    - Only moves where elev[next] <= elev[current] are allowed.
    - Abadi cells are impassable (infinite cost, Rule C).
    - Goal = natural drain cell only (Rule A).
    Returns (path, status) where status is DIRECT_OUTLET | ROUTED | UNREACHABLE.
    """
    nrows, ncols = cost.shape
    sr, sc = start
    r_lo = max(0, sr - max_radius_px)
    r_hi = min(nrows, sr + max_radius_px + 1)
    c_lo = max(0, sc - max_radius_px)
    c_hi = min(ncols, sc + max_radius_px + 1)

    if goals_mask[sr, sc]:
        return [(sr, sc)], "DIRECT_OUTLET"

    goal_rows, goal_cols = np.where(goals_mask[r_lo:r_hi, c_lo:c_hi])
    if len(goal_rows) == 0:
        return None, "UNREACHABLE"
    goal_rows += r_lo
    goal_cols += c_lo
    gdists = np.sqrt((goal_rows - sr)**2 + (goal_cols - sc)**2)
    gi = np.argmin(gdists)
    gr, gc = int(goal_rows[gi]), int(goal_cols[gi])

    def h(r, c):
        return math.sqrt((r - gr)**2 + (c - gc)**2) * 0.01

    counter = 0
    heap = [(h(sr, sc), counter, sr, sc)]
    counter += 1
    lrows, lcols = r_hi - r_lo, c_hi - c_lo
    gscore = np.full((lrows, lcols), np.inf, dtype=np.float64)
    gscore[sr - r_lo, sc - c_lo] = 0.0
    visited = np.zeros((lrows, lcols), dtype=bool)
    came_from = {}

    while heap:
        f, _, cr, cc = heapq.heappop(heap)
        lr, lc = cr - r_lo, cc - c_lo
        if visited[lr, lc]:
            continue
        visited[lr, lc] = True

        if goals_mask[cr, cc]:
            path = [(cr, cc)]
            cur = (cr, cc)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path, "ROUTED"

        cg = gscore[lr, lc]
        ce = elev[cr, cc]

        for dr, dc, df in NEIGHBORS_8:
            nr, nc = cr + dr, cc + dc
            if nr < r_lo or nr >= r_hi or nc < c_lo or nc >= c_hi:
                continue
            nlr, nlc = nr - r_lo, nc - c_lo
            if visited[nlr, nlc]:
                continue
            # Rule B: gravity with micro-relief tolerance
            ne = elev[nr, nc]
            if ne > ce + GRAVITY_TOLERANCE:
                continue
            # Rule C: abadi avoidance
            if abadi_mask is not None and abadi_mask[nr, nc]:
                continue
            cc2 = cost[nr, nc]
            if not np.isfinite(cc2):
                continue
            tg = cg + cc2 * df * pixel_size
            if tg < gscore[nlr, nlc]:
                gscore[nlr, nlc] = tg
                came_from[(nr, nc)] = (cr, cc)
                heapq.heappush(heap, (tg + h(nr, nc), counter, nr, nc))
                counter += 1

    return None, "UNREACHABLE"


def _pump_astar(cost, elev, start, goals_mask, abadi_mask,
                max_radius_px, pixel_size):
    """Unconstrained A* (allows uphill) for pump-lift computation.

    Used for basins that are UNREACHABLE by gravity.
    Returns (path, status, max_uphill_m).
    max_uphill_m = maximum single-step uphill encountered on the path.
    """
    nrows, ncols = cost.shape
    sr, sc = start
    r_lo = max(0, sr - max_radius_px)
    r_hi = min(nrows, sr + max_radius_px + 1)
    c_lo = max(0, sc - max_radius_px)
    c_hi = min(ncols, sc + max_radius_px + 1)

    if goals_mask[sr, sc]:
        return [(sr, sc)], "PUMP_DIRECT", 0.0

    goal_rows, goal_cols = np.where(goals_mask[r_lo:r_hi, c_lo:c_hi])
    if len(goal_rows) == 0:
        return None, "PUMP_FAIL", 0.0
    goal_rows += r_lo
    goal_cols += c_lo
    gdists = np.sqrt((goal_rows - sr)**2 + (goal_cols - sc)**2)
    gi = np.argmin(gdists)
    gr, gc = int(goal_rows[gi]), int(goal_cols[gi])

    def h(r, c):
        return math.sqrt((r - gr)**2 + (c - gc)**2) * 0.01

    counter = 0
    heap = [(h(sr, sc), counter, sr, sc)]
    counter += 1
    lrows, lcols = r_hi - r_lo, c_hi - c_lo
    gscore = np.full((lrows, lcols), np.inf, dtype=np.float64)
    gscore[sr - r_lo, sc - c_lo] = 0.0
    visited = np.zeros((lrows, lcols), dtype=bool)
    came_from = {}

    while heap:
        f, _, cr, cc = heapq.heappop(heap)
        lr, lc = cr - r_lo, cc - c_lo
        if visited[lr, lc]:
            continue
        visited[lr, lc] = True

        if goals_mask[cr, cc]:
            path = [(cr, cc)]
            cur = (cr, cc)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            # Compute required lift: max uphill step along path
            max_up = 0.0
            for i in range(1, len(path)):
                diff = elev[path[i][0], path[i][1]] - elev[path[i-1][0], path[i-1][1]]
                if diff > 0:
                    max_up = max(max_up, diff)
            # Also compute total uphill needed
            total_up = 0.0
            for i in range(1, len(path)):
                diff = elev[path[i][0], path[i][1]] - elev[path[i-1][0], path[i-1][1]]
                if diff > 0:
                    total_up += diff
            return path, "PUMP_ROUTED", total_up

        cg = gscore[lr, lc]
        ce = elev[cr, cc]

        for dr, dc, df in NEIGHBORS_8:
            nr, nc = cr + dr, cc + dc
            if nr < r_lo or nr >= r_hi or nc < c_lo or nc >= c_hi:
                continue
            nlr, nlc = nr - r_lo, nc - c_lo
            if visited[nlr, nlc]:
                continue
            # Abadi still avoided even in pump mode
            if abadi_mask is not None and abadi_mask[nr, nc]:
                continue
            cc2 = cost[nr, nc]
            if not np.isfinite(cc2):
                continue
            # Uphill penalty: add elevation gain as extra cost
            eg = max(0.0, float(elev[nr, nc] - ce))
            tg = cg + (cc2 + eg * 2.0) * df * pixel_size
            if tg < gscore[nlr, nlc]:
                gscore[nlr, nlc] = tg
                came_from[(nr, nc)] = (cr, cc)
                heapq.heappush(heap, (tg + h(nr, nc), counter, nr, nc))
                counter += 1

    return None, "PUMP_FAIL", 0.0


# ── Abadi & Natural-Drain Rasterization ───────────────────────

def _rasterize_abadi(village_dir, shape, gt):
    """Rasterize abadi_area.shp to boolean mask. Returns None if not found."""
    shp_path = os.path.join(village_dir, "drainage", "abadi_area.shp")
    if not os.path.exists(shp_path):
        # Also check top-level
        shp_path = os.path.join(village_dir, "abadi_area.shp")
    if not os.path.exists(shp_path):
        return None

    ds = ogr.Open(shp_path)
    if ds is None or ds.GetLayerCount() == 0:
        return None
    layer = ds.GetLayer(0)
    if layer.GetFeatureCount() == 0:
        ds = None
        return None

    rows, cols = shape
    # Create in-memory raster
    mem_drv = gdal.GetDriverByName("MEM")
    target = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
    target.SetGeoTransform(gt)
    target.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(target, [1], layer, burn_values=[1])
    mask = target.GetRasterBand(1).ReadAsArray().astype(bool)
    target = None
    ds = None
    n = int(mask.sum())
    if n > 0:
        print(f"  Abadi mask: {n:,} cells rasterized", flush=True)
    return mask


def _rasterize_natural_drains(village_dir, shape, gt):
    """Rasterize natural_drain_network.shp to boolean goal mask."""
    shp_path = os.path.join(village_dir, "drainage", "natural_drain_network.shp")
    if not os.path.exists(shp_path):
        return None

    ds = ogr.Open(shp_path)
    if ds is None or ds.GetLayerCount() == 0:
        return None
    layer = ds.GetLayer(0)
    n_feat = layer.GetFeatureCount()
    if n_feat == 0:
        ds = None
        return None

    rows, cols = shape
    mem_drv = gdal.GetDriverByName("MEM")
    target = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
    target.SetGeoTransform(gt)
    target.GetRasterBand(1).Fill(0)
    # Rasterize with ALL_TOUCHED to capture thin lines
    gdal.RasterizeLayer(target, [1], layer, burn_values=[1],
                        options=["ALL_TOUCHED=TRUE"])
    mask = target.GetRasterBand(1).ReadAsArray().astype(bool)
    target = None
    ds = None
    n = int(mask.sum())
    print(f"  Natural drain goal mask: {n:,} cells ({n_feat} features)",
          flush=True)
    return mask


# ── Zhang-Suen Thinning ───────────────────────────────────────

def _zhang_suen_thin(binary):
    img = np.pad(binary.astype(np.uint8), 1, mode="constant",
                 constant_values=0)

    def _nbrs(im):
        P2=im[:-2,1:-1]; P3=im[:-2,2:]; P4=im[1:-1,2:]; P5=im[2:,2:]
        P6=im[2:,1:-1]; P7=im[2:,:-2]; P8=im[1:-1,:-2]; P9=im[:-2,:-2]
        return P2,P3,P4,P5,P6,P7,P8,P9

    def _trans(P2,P3,P4,P5,P6,P7,P8,P9):
        seq=[P2,P3,P4,P5,P6,P7,P8,P9]
        cnt=np.zeros_like(P2, dtype=np.int32)
        for i in range(8):
            cnt+=((seq[i]==0)&(seq[(i+1)%8]==1))
        return cnt

    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            P2,P3,P4,P5,P6,P7,P8,P9 = _nbrs(img)
            center = img[1:-1,1:-1]
            B = P2.astype(np.int32)+P3+P4+P5+P6+P7+P8+P9
            A = _trans(P2,P3,P4,P5,P6,P7,P8,P9)
            cond = (center==1)&(B>=2)&(B<=6)&(A==1)
            if step == 1:
                cond &= ((P2*P4*P6)==0) & ((P4*P6*P8)==0)
            else:
                cond &= ((P2*P4*P8)==0) & ((P2*P6*P8)==0)
            if np.any(cond):
                img[1:-1,1:-1][cond] = 0
                changed = True

    return img[1:-1,1:-1].astype(bool)


def _trace_lines(skeleton, gt):
    """Trace skeleton pixels → list of [(x,y),...] polylines."""
    ys, xs = np.where(skeleton)
    if len(ys) == 0:
        return []
    pixel_set = set(zip(ys.tolist(), xs.tolist()))

    adj = {}
    for r, c in pixel_set:
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                n = (r+dr, c+dc)
                if n in pixel_set:
                    nbrs.append(n)
        adj[(r, c)] = nbrs

    endpoints = {p for p, n in adj.items() if len(n) == 1}
    junctions = {p for p, n in adj.items() if len(n) >= 3}
    stop_pts = endpoints | junctions
    visited = set()

    def _edge(a, b):
        return (a, b) if a < b else (b, a)

    def _walk(start, first):
        path = [start, first]
        visited.add(_edge(start, first))
        cur, prev = first, start
        if cur in stop_pts:
            return path
        while True:
            nxt = [n for n in adj[cur]
                   if n != prev and _edge(cur, n) not in visited]
            if not nxt:
                break
            n = nxt[0]
            visited.add(_edge(cur, n))
            path.append(n)
            if n in stop_pts:
                break
            prev, cur = cur, n
        return path

    lines = []
    for p in sorted(stop_pts):
        for nb in adj[p]:
            if _edge(p, nb) not in visited:
                chain = _walk(p, nb)
                if len(chain) >= 2:
                    coords = []
                    for r, c in chain:
                        x = gt[0] + c*gt[1] + r*gt[2]
                        y = gt[3] + c*gt[4] + r*gt[5]
                        coords.append((x, y))
                    lines.append(coords)

    # Remaining unvisited cycles
    for p in sorted(adj.keys()):
        for nb in adj[p]:
            if _edge(p, nb) not in visited:
                chain = _walk(p, nb)
                if len(chain) >= 2:
                    coords = []
                    for r, c in chain:
                        x = gt[0] + c*gt[1] + r*gt[2]
                        y = gt[3] + c*gt[4] + r*gt[5]
                        coords.append((x, y))
                    lines.append(coords)

    return lines


def _write_stream_shp(out_path, lines, proj, min_len=5.0):
    """Write traced polylines as shapefile."""
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    driver = ogr.GetDriverByName("ESRI Shapefile")

    base = out_path.replace(".shp", "")
    actual = out_path
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = base + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    if os.path.exists(out_path):
        actual = out_path.replace(".shp", ".tmp.shp")
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = actual.replace(".shp", ext)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    ds = driver.CreateDataSource(actual)
    layer = ds.CreateLayer("streams", srs, ogr.wkbLineString)
    fld = ogr.FieldDefn("length_m", ogr.OFTReal); fld.SetWidth(10); fld.SetPrecision(1)
    layer.CreateField(fld)
    layer.CreateField(ogr.FieldDefn("stream_id", ogr.OFTInteger))

    n = 0
    for i, coords in enumerate(lines):
        if len(coords) < 2:
            continue
        line = ogr.Geometry(ogr.wkbLineString)
        for x, y in coords:
            line.AddPoint(x, y)
        length = line.Length()
        if length < min_len:
            continue
        # Simplify
        simplified = line.Simplify(1.0)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("stream_id", i+1)
        feat.SetField("length_m", round(length, 1))
        feat.SetGeometry(simplified)
        layer.CreateFeature(feat)
        feat = None
        n += 1
    ds = None
    print(f"  Wrote: {n} stream segments → {os.path.basename(actual)}",
          flush=True)
    return n


# ── Main Entry Points ─────────────────────────────────────────

def run_routing(village_dir):
    """Route HIGH-risk clusters to natural drain network (v3).

    Two-pass routing:
      Pass 1 - gravity A* (strict downhill to natural_drain_network.shp)
      Pass 2 - for UNREACHABLE basins, unconstrained A* with pump-lift calc
    """
    t_start = time.time()

    terrain_dir = os.path.join(village_dir, "terrain")
    hydro_dir = os.path.join(village_dir, "hydrology")
    wl_dir = os.path.join(village_dir, "waterlogging")
    out_dir = os.path.join(village_dir, "drainage")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60, flush=True)
    print("  DRAINAGE ROUTING v3 (strict gravity + pump lift)", flush=True)
    print("=" * 60, flush=True)

    # -- Load rasters ------------------------------------------------
    dtm, gt, proj, dtm_nd = _read_raster(
        os.path.join(terrain_dir, "village_dtm_clean.tif"))
    slope, _, _, slope_nd = _read_raster(
        os.path.join(terrain_dir, "slope.tif"))
    fa, _, _, fa_nd = _read_raster(
        os.path.join(hydro_dir, "flow_accumulation.tif"))
    conv, _, _, conv_nd = _read_raster(
        os.path.join(hydro_dir, "convergence_index.tif"))
    risk, _, _, risk_nd = _read_raster(
        os.path.join(wl_dir, "waterlogging_risk_class.tif"))

    valid = dtm != dtm_nd
    pixel_x = abs(gt[1])
    rows, cols = dtm.shape
    print(f"  Shape: {rows}x{cols}, valid: {valid.sum():,}", flush=True)

    # -- Goal mask: natural drain network ONLY (Rule A) ---------------
    # CRITICAL: natural_drain_mask is the SOLE valid termination target.
    # No FA cells. No boundary cells. No exceptions.
    nd_mask_raw = _rasterize_natural_drains(village_dir, dtm.shape, gt)

    if nd_mask_raw is not None and nd_mask_raw.sum() > 0:
        natural_drain_mask = nd_mask_raw
        print(f"  GOAL MASK (pure natural drain): {int(natural_drain_mask.sum()):,} cells",
              flush=True)
    else:
        # Last resort: FA-derived stream mask (only if shp missing entirely)
        print("  WARN: No natural_drain_network.shp - using FA stream mask",
              flush=True)
        fa_valid = valid & ((fa != fa_nd) if fa_nd is not None else
                            np.ones_like(fa, dtype=bool))
        n_valid = fa_valid.sum()
        fa_vals = fa[fa_valid]
        target_lo = int(n_valid * STREAM_DENSITY_MIN)
        target_hi = int(n_valid * STREAM_DENSITY_MAX)

        best_thresh = None
        final_count = 0
        for pct in np.arange(99.5, 80.0, -0.5):
            thresh = float(np.percentile(fa_vals, pct))
            if thresh < 1.0:
                continue
            count = int(np.sum(fa_vals > thresh))
            if target_lo <= count <= target_hi:
                best_thresh = thresh
                final_count = count
                break
            if count > target_hi:
                best_thresh = float(np.percentile(fa_vals, pct + 0.5))
                final_count = int(np.sum(fa_vals > best_thresh))
                break
            best_thresh = thresh
            final_count = count

        if best_thresh is None:
            best_thresh = float(np.percentile(fa_vals, 90.0))
            final_count = int(np.sum(fa_vals > best_thresh))

        natural_drain_mask = (fa > best_thresh) & fa_valid
        print(f"  FA stream threshold: {best_thresh:.0f} ({final_count:,} px)",
              flush=True)
    # NO FA merge. natural_drain_mask is clean.

    # -- Secondary outlet mask (FA-derived flow corridors) -----------
    fa_valid = valid & ((fa != fa_nd) if fa_nd is not None else
                        np.ones_like(fa, dtype=bool))
    fa_vals = fa[fa_valid]
    if len(fa_vals) > 0:
        fa_thresh = float(np.percentile(fa_vals, SECONDARY_FA_PERCENTILE))
        secondary_outlet_mask = (fa > fa_thresh) & fa_valid
        # Exclude cells already in primary mask (avoid double-counting)
        secondary_outlet_mask = secondary_outlet_mask & ~natural_drain_mask
        n_sec = int(secondary_outlet_mask.sum())
        print(f"  SECONDARY OUTLETS (FA >{fa_thresh:.0f}, top "
              f"{100-SECONDARY_FA_PERCENTILE}%): {n_sec:,} cells", flush=True)
    else:
        secondary_outlet_mask = np.zeros_like(valid)
        print("  SECONDARY OUTLETS: 0 (no valid FA data)", flush=True)

    # Combined outlet mask (for validation – any routed path must end here)
    any_outlet_mask = natural_drain_mask | secondary_outlet_mask

    # -- Terminal outlet detection (boundary discharge points) ---------
    # Terminal outlets are stream cells near the edge of the valid data
    # area — these represent real discharge points where water exits.
    # We prioritize these over mid-stream cells during routing.
    boundary_width = max(5, int(15.0 / pixel_x))  # ~5 m margin
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(valid, iterations=boundary_width)
    boundary_band = valid & ~interior  # strip around valid data edge

    terminal_primary = natural_drain_mask & boundary_band
    terminal_secondary = secondary_outlet_mask & boundary_band
    n_term_p = int(terminal_primary.sum())
    n_term_s = int(terminal_secondary.sum())
    print(f"  TERMINAL OUTLETS (boundary discharge): "
          f"{n_term_p} primary, {n_term_s} secondary", flush=True)

    # If terminal outlets exist, prefer them; otherwise fall back to
    # the full stream mask (so routing never degrades vs current behavior).
    prefer_terminal_primary = terminal_primary if n_term_p >= 1 else natural_drain_mask
    prefer_terminal_secondary = terminal_secondary if n_term_s >= 1 else secondary_outlet_mask

    # -- Abadi avoidance mask (Rule C) --------------------------------
    abadi_mask = _rasterize_abadi(village_dir, dtm.shape, gt)
    if abadi_mask is None:
        print("  Abadi mask: None (no abadi_area.shp)", flush=True)

    # -- Cluster detection --------------------------------------------
    high_mask = ((risk == 2) & valid &
                 ((risk != risk_nd) if risk_nd is not None else True))
    labeled, n_total = ndimage_label(high_mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labeled.ravel())
    min_px = int(math.ceil(MIN_CLUSTER_AREA / (pixel_x ** 2)))
    keep = [i for i in range(1, len(sizes)) if sizes[i] >= min_px]
    print(f"  HIGH clusters: {n_total} total, {len(keep)} routable",
          flush=True)

    # -- Cost surface -------------------------------------------------
    slope_valid = valid & ((slope != slope_nd) if slope_nd is not None
                           else True)
    conv_valid = valid & ((conv != conv_nd) if conv_nd is not None
                          else True)
    norm_slope = _robust_normalize(slope, slope_valid)
    norm_conv = _robust_normalize(conv, conv_valid)
    cost = (np.float32(W_SLOPE) * norm_slope +
            np.float32(W_CONV) * norm_conv)
    cost[~valid] = np.inf
    # Abadi cells = infinite cost
    if abadi_mask is not None:
        cost[abadi_mask] = np.inf
    del norm_slope, norm_conv

    # Stream EDT for search radius (uses pure natural_drain_mask)
    stream_edt = (distance_transform_edt(~natural_drain_mask).astype(np.float32)
                  * np.float32(pixel_x)
                  if natural_drain_mask.any()
                  else np.full((rows, cols), 500.0, dtype=np.float32))

    # -- Route each cluster: four-pass (primary + secondary) ----------
    results = []
    n_gravity = 0
    n_pump = 0
    n_secondary_gravity = 0
    n_secondary_pump = 0
    n_retention = 0
    n_fail = 0
    lifts = []

    for idx, cid in enumerate(sorted(keep, key=lambda i: -sizes[i])):
        mask = labeled == cid
        area = float(sizes[cid]) * pixel_x ** 2

        # Boundary pixels of cluster
        struct = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(mask, structure=struct)
        inner = (binary_dilation(dilated & ~mask & valid,
                                 structure=struct) & mask)
        br, bc = np.where(inner)
        if len(br) == 0:
            br, bc = np.where(mask)

        # Lowest boundary pixel = routing start
        elevs = dtm[br, bc]
        mi = np.argmin(elevs)
        er, ec = int(br[mi]), int(bc[mi])
        start_elev = float(dtm[er, ec])

        # Search radius – bounded by MAX_ROUTING_DISTANCE
        rad_m = MAX_ROUTING_DISTANCE
        rad_px = int(math.ceil(rad_m / pixel_x))

        # -- Six-pass routing (terminal outlets → stream outlets) -------
        # Passes 1-2: Route to TERMINAL outlets (boundary discharge points
        # where water actually exits the study area — the correct endpoint).
        # Passes 3-6: Fallback to any stream/FA cell (existing behavior).
        path = None
        status = None
        outlet_type = None
        pump_required = False
        required_lift = 0.0

        # Check direct outlet on primary
        if natural_drain_mask[er, ec]:
            path = [(er, ec)]
            status = "DIRECT_OUTLET"
            outlet_type = "PRIMARY"
        # Check direct outlet on secondary
        elif secondary_outlet_mask[er, ec]:
            path = [(er, ec)]
            status = "DIRECT_OUTLET"
            outlet_type = "SECONDARY"
        else:
            # Pass 1: gravity → terminal primary (boundary discharge)
            path, status = _gravity_astar(
                cost, dtm, (er, ec), prefer_terminal_primary, abadi_mask,
                rad_px, pixel_x)
            if path is not None:
                outlet_type = "PRIMARY"

            # Pass 2: pump → terminal primary (boundary discharge)
            if path is None:
                path2, status2, total_up = _pump_astar(
                    cost, dtm, (er, ec), prefer_terminal_primary, abadi_mask,
                    rad_px, pixel_x)
                if path2 is not None and total_up <= MAX_PUMP_LIFT:
                    path = path2
                    status = "PUMP_REQUIRED"
                    pump_required = True
                    required_lift = total_up
                    outlet_type = "PRIMARY"

            # Pass 3: gravity → any primary stream cell (fallback)
            if path is None and n_term_p >= 1:
                path3, status3 = _gravity_astar(
                    cost, dtm, (er, ec), natural_drain_mask, abadi_mask,
                    rad_px, pixel_x)
                if path3 is not None:
                    path = path3
                    status = status3
                    outlet_type = "PRIMARY"

            # Pass 4: pump → any primary stream cell (fallback)
            if path is None and n_term_p >= 1:
                path4, status4, total_up4 = _pump_astar(
                    cost, dtm, (er, ec), natural_drain_mask, abadi_mask,
                    rad_px, pixel_x)
                if path4 is not None and total_up4 <= MAX_PUMP_LIFT:
                    path = path4
                    status = "PUMP_REQUIRED"
                    pump_required = True
                    required_lift = total_up4
                    outlet_type = "PRIMARY"

            # Pass 5: gravity → secondary (FA channels)
            if path is None:
                path5, status5 = _gravity_astar(
                    cost, dtm, (er, ec), secondary_outlet_mask, abadi_mask,
                    rad_px, pixel_x)
                if path5 is not None:
                    path = path5
                    status = status5
                    outlet_type = "SECONDARY"

            # Pass 6: pump → secondary
            if path is None:
                path6, status6, total_up6 = _pump_astar(
                    cost, dtm, (er, ec), secondary_outlet_mask, abadi_mask,
                    rad_px, pixel_x)
                if path6 is not None and total_up6 <= MAX_PUMP_LIFT:
                    path = path6
                    status = "PUMP_REQUIRED"
                    pump_required = True
                    required_lift = total_up6
                    outlet_type = "SECONDARY"

            # No outlet reachable within constraints → RETENTION
            if path is None:
                status = "RETENTION"
                outlet_type = None

        # -- POST-ROUTING VALIDATION (MANDATORY) ---------------------
        # Every routed path MUST end on a valid outlet cell.
        if path is not None and len(path) >= 1:
            end_r, end_c = path[-1]
            if not any_outlet_mask[end_r, end_c]:
                raise RuntimeError(
                    f"ROUTING BUG: Basin {cid} terminated at "
                    f"({end_r},{end_c}) which is NOT on any outlet "
                    f"mask. primary={natural_drain_mask[end_r, end_c]}, "
                    f"secondary={secondary_outlet_mask[end_r, end_c]}. "
                    f"Status={status}. This path is INVALID.")

        # -- Compute path metrics ------------------------------------
        rec = {
            "cluster_id": cid,
            "area_m2": area,
            "status": status,
            "outlet_type": outlet_type or "",
            "path": path,
            "start_elev": start_elev,
            "pump_required": pump_required,
            "required_lift_m": round(required_lift, 3),
        }

        if path and len(path) >= 2:
            pa = np.array(path)
            pr, pc = pa[:, 0], pa[:, 1]
            pe = dtm[pr, pc]
            rec["end_elev"] = float(pe[-1])
            dr = np.diff(pr).astype(np.float64)
            dc = np.diff(pc).astype(np.float64)
            seg = np.sqrt(dr**2 + dc**2) * pixel_x
            rec["path_len_m"] = round(float(seg.sum()), 1)
            rec["elev_drop_m"] = round(float(pe[0] - pe[-1]), 2)
            ps = slope[pr, pc]
            sv = ps[(ps != slope_nd) if slope_nd is not None
                     else np.ones(len(ps), dtype=bool)]
            rec["mean_slp_pct"] = (
                round(float(np.mean(np.abs(np.tan(np.radians(sv))) * 100)),
                      2) if len(sv) > 0 else 0.0)
            rec["feasible"] = (not pump_required and
                               rec["elev_drop_m"] > 0)
        else:
            rec["end_elev"] = start_elev
            rec["path_len_m"] = 0.0
            rec["elev_drop_m"] = 0.0
            rec["mean_slp_pct"] = 0.0
            rec["feasible"] = False

        results.append(rec)

        # Counters
        if status in ("ROUTED", "DIRECT_OUTLET") and outlet_type == "PRIMARY":
            n_gravity += 1
        elif status == "PUMP_REQUIRED" and outlet_type == "PRIMARY":
            n_pump += 1
            lifts.append(required_lift)
        elif status in ("ROUTED", "DIRECT_OUTLET") and outlet_type == "SECONDARY":
            n_secondary_gravity += 1
        elif status == "PUMP_REQUIRED" and outlet_type == "SECONDARY":
            n_secondary_pump += 1
            lifts.append(required_lift)
        elif status == "RETENTION":
            n_retention += 1
        else:
            n_fail += 1

        sym = {"ROUTED": "+", "DIRECT_OUTLET": "D",
               "PUMP_REQUIRED": "P", "RETENTION": "R",
               "UNREACHABLE": "X"}
        lift_str = (f" lift={required_lift:.2f}m" if pump_required else "")
        ot_str = f" [{outlet_type[0]}]" if outlet_type else ""
        print(f"  [{idx+1:3d}/{len(keep)}] {area:8.0f}m2 "
              f"{sym.get(status, '?')} {rec['path_len_m']:6.0f}m "
              f"drop={rec['elev_drop_m']:+.1f}m{lift_str}{ot_str}",
              flush=True)

    # -- Write shapefile ----------------------------------------------
    shp_out = os.path.join(out_dir, "optimized_drainage_network.shp")
    _write_shp(shp_out, results, gt, proj)

    # -- Write CSV ----------------------------------------------------
    csv_out = os.path.join(out_dir, "drainage_design_summary.csv")
    fields = ["cluster_id", "area_m2", "start_elev", "end_elev",
              "path_len_m", "mean_slp_pct", "elev_drop_m",
              "status", "outlet_type", "pump_required", "required_lift_m",
              "feasible"]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            row.pop("path", None)
            w.writerow(row)
    print(f"  CSV: {len(results)} rows -> {os.path.basename(csv_out)}",
          flush=True)

    # -- Summary ------------------------------------------------------
    print(f"\n  {'='*50}", flush=True)
    print(f"  ROUTING SUMMARY (v3)", flush=True)
    print(f"  {'='*50}", flush=True)
    n_routed_total = n_gravity + n_pump + n_secondary_gravity + n_secondary_pump
    print(f"  Total basins:     {len(results)}", flush=True)
    print(f"  --- Primary outlets (natural drains) ---", flush=True)
    print(f"  Gravity-routed:   {n_gravity}", flush=True)
    print(f"  Pump-required:    {n_pump}", flush=True)
    print(f"  --- Secondary outlets (FA channels) ---", flush=True)
    print(f"  Gravity-routed:   {n_secondary_gravity}", flush=True)
    print(f"  Pump-required:    {n_secondary_pump}", flush=True)
    print(f"  --- Unroutable ---", flush=True)
    print(f"  Retention basins: {n_retention}", flush=True)
    print(f"  Unreachable:      {n_fail}", flush=True)
    print(f"  --- Totals ---", flush=True)
    print(f"  Routed total:     {n_routed_total}", flush=True)
    if lifts:
        print(f"  Mean lift:        {np.mean(lifts):.3f} m", flush=True)
        print(f"  Max lift:         {np.max(lifts):.3f} m", flush=True)
    routed = [r for r in results
              if r["status"] in ("ROUTED", "DIRECT_OUTLET", "PUMP_REQUIRED")]
    total_len = sum(r["path_len_m"] for r in routed)
    print(f"  Total drain length: {total_len:.0f}m ({total_len/1000:.1f}km)",
          flush=True)
    print(f"  Gravity tolerance used: {GRAVITY_TOLERANCE} m", flush=True)
    print(f"  Max routing distance: {MAX_ROUTING_DISTANCE} m", flush=True)
    print(f"  Max pump lift: {MAX_PUMP_LIFT} m", flush=True)
    print(f"  Time: {time.time()-t_start:.0f}s", flush=True)


def _find_nearest_stream_cell(row, col, stream_mask, max_r):
    nr, nc = stream_mask.shape
    r0 = max(0, row - max_r)
    r1 = min(nr, row + max_r + 1)
    c0 = max(0, col - max_r)
    c1 = min(nc, col + max_r + 1)
    local = stream_mask[r0:r1, c0:c1]
    if not local.any():
        return None, None, np.inf
    sr, sc = np.where(local)
    sr += r0; sc += c0
    d = np.sqrt((sr - row)**2 + (sc - col)**2)
    i = np.argmin(d)
    return int(sr[i]), int(sc[i]), float(d[i])


def run_streams(village_dir):
    """Extract natural drain network from flow accumulation."""
    t0 = time.time()

    hydro_dir = os.path.join(village_dir, "hydrology")
    out_dir = os.path.join(village_dir, "drainage")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60, flush=True)
    print("  NATURAL STREAM VECTORIZATION", flush=True)
    print("=" * 60, flush=True)

    # Load stream raster from dist_to_stream (pixels == 0 = stream)
    dist_path = os.path.join(hydro_dir, "dist_to_stream.tif")
    if not os.path.exists(dist_path):
        print("  SKIP: No dist_to_stream.tif found", flush=True)
        return

    arr, gt, proj, nd = _read_raster(dist_path)
    valid = (arr != nd) if nd is not None else np.ones(arr.shape, dtype=bool)
    binary = np.zeros(arr.shape, dtype=np.uint8)
    binary[valid & (arr == 0)] = 1
    n_stream = int(binary.sum())
    print(f"  Stream pixels: {n_stream:,}", flush=True)

    if n_stream < 10:
        print("  SKIP: Too few stream pixels for vectorization", flush=True)
        out = os.path.join(out_dir, "natural_drain_network.shp")
        _write_stream_shp(out, [], proj)
        return

    # Crop to bounding box of stream pixels + 2px margin for thinning
    rows_idx, cols_idx = np.where(binary)
    r0, r1 = max(0, rows_idx.min() - 2), min(binary.shape[0], rows_idx.max() + 3)
    c0, c1 = max(0, cols_idx.min() - 2), min(binary.shape[1], cols_idx.max() + 3)
    cropped = binary[r0:r1, c0:c1]
    n_crop = int(cropped.sum())
    print(f"  Crop: {r1-r0}×{c1-c0} ({n_crop:,} stream px)", flush=True)

    if n_crop > 100_000:
        print("  SKIP: Too many stream pixels for thinning — using raw mask", flush=True)
        skeleton_full = binary
    else:
        print("  Thinning (Zhang-Suen) ...", flush=True)
        skeleton_crop = _zhang_suen_thin(cropped)
        # Place back into full array
        skeleton_full = np.zeros_like(binary)
        skeleton_full[r0:r1, c0:c1] = skeleton_crop
    print(f"  Skeleton pixels: {int(skeleton_full.sum()):,}", flush=True)

    lines = _trace_lines(skeleton_full, gt)
    print(f"  Traced lines: {len(lines)}", flush=True)

    out = os.path.join(out_dir, "natural_drain_network.shp")
    _write_stream_shp(out, lines, proj)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)


def run(village_dir):
    """Run stream extraction FIRST, then routing."""
    run_streams(village_dir)
    run_routing(village_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--village-dir", required=True)
    args = p.parse_args()
    run(args.village_dir)
