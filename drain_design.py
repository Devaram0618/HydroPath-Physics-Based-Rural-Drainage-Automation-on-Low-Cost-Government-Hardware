"""
engine/drain_design.py — Drain Design Parameters (Rational Method)
====================================================================
Computes engineering design parameters for each proposed drain route
using the Rational Method (Q = C × I × A).

Inputs:
    terrain/village_dtm_clean.tif, terrain/slope.tif
    hydrology/flow_accumulation.tif
    waterlogging/waterlogging_risk_class.tif
    drainage/drainage_design_summary.csv
    hydrology/watersheds.tif  (optional, from watershed module)

Outputs (to village_dir/drainage/):
    drain_design_parameters.csv   — per-drain engineering specs
    drain_design_summary.json     — aggregated design metrics

Design outputs per drain:
    catchment_area_ha     — upstream catchment contributing to this drain
    runoff_coefficient    — C (dimensionless, terrain-adaptive)
    rainfall_intensity    — I (mm/hr, from IDF assumptions)
    peak_discharge_m3s    — Q = C × I × A (m³/s)
    pipe_diameter_mm      — Manning's equation pipe sizing
    trench_depth_m        — recommended excavation depth
    trench_width_m        — recommended trench width
    bed_slope             — drain gradient (m/m)
    manning_velocity_ms   — flow velocity from Manning's equation

Zero impact on waterlogging model or scoring engine.
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
from osgeo import gdal
from scipy.ndimage import label as ndimage_label

gdal.UseExceptions()

NODATA = -9999.0

# ── Design Constants ───────────────────────────────────────────
# Rainfall intensity (mm/hr) — 10-year return period, 1-hour duration
# Default for rural India; overridable via config
DEFAULT_RAINFALL_INTENSITY = 75.0   # mm/hr (moderate monsoon)
HEAVY_RAINFALL_INTENSITY = 120.0    # mm/hr (severe monsoon)

# Manning's roughness coefficient
MANNING_N_EARTHEN = 0.025           # unlined earthen channel
MANNING_N_LINED = 0.015             # concrete/brick-lined channel

# Minimum design parameters
MIN_PIPE_DIAMETER_MM = 150          # minimum drain diameter
MAX_PIPE_DIAMETER_MM = 1200         # maximum practical diameter
MIN_BED_SLOPE = 0.001              # 1:1000 minimum gradient
MIN_TRENCH_DEPTH_M = 0.45          # minimum depth (45 cm)
MAX_TRENCH_DEPTH_M = 2.5           # maximum practical depth
FREEBOARD_RATIO = 0.20             # 20% freeboard above design flow

# Runoff coefficients by terrain type
RUNOFF_COEFFICIENTS = {
    "COASTAL_LOWLAND": 0.70,       # saturated, high water table
    "FLAT": 0.55,                   # typical agricultural flat
    "MODERATE": 0.45,               # moderate slopes, partial infiltration
    "HILLY": 0.35,                  # steeper, faster runoff but less ponding
}

# Trapezoidal channel design constants
SIDE_SLOPE = 1.5                    # horizontal:vertical (1.5:1)


# ── Raster I/O ─────────────────────────────────────────────────

def _read_raster(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        return None, None, None, None
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return arr, gt, proj, nd


# ── Helper Functions ───────────────────────────────────────────

def _classify_terrain(dtm, slope, valid):
    """Classify terrain type from DEM and slope statistics."""
    dtm_valid = dtm[valid]
    slope_valid = slope[valid]
    mean_elev = float(np.nanmean(dtm_valid))
    min_elev = float(np.nanmin(dtm_valid))
    mean_slope = float(np.nanmean(slope_valid))

    if mean_elev < 5.0 or min_elev < 0:
        return "COASTAL_LOWLAND"
    elif mean_slope < 3.0:
        return "FLAT"
    elif mean_slope < 8.0:
        return "MODERATE"
    else:
        return "HILLY"


def _compute_catchment_area(fa_arr, dtm_arr, valid, cluster_mask,
                            pixel_area_m2, gt):
    """Estimate catchment area contributing to a cluster.

    Uses flow accumulation at the pour point (lowest pixel) of the cluster.
    """
    # Find pour point: lowest elevation in cluster
    cluster_dtm = np.where(cluster_mask & valid, dtm_arr, np.inf)
    if not np.any(np.isfinite(cluster_dtm)):
        return float(cluster_mask.sum()) * pixel_area_m2 / 10000.0

    r_min, c_min = np.unravel_index(np.argmin(cluster_dtm), cluster_dtm.shape)

    # Flow accumulation at pour point = upstream pixel count
    if fa_arr is not None:
        fa_val = float(fa_arr[r_min, c_min])
        if np.isfinite(fa_val) and fa_val > 0:
            catchment_px = fa_val
        else:
            catchment_px = float(cluster_mask.sum())
    else:
        catchment_px = float(cluster_mask.sum())

    return catchment_px * pixel_area_m2 / 10000.0  # hectares


def _rational_method(C, I_mm_hr, A_ha):
    """Rational Method: Q = C × I × A / 360

    Args:
        C: runoff coefficient (dimensionless)
        I_mm_hr: rainfall intensity (mm/hr)
        A_ha: catchment area (hectares)

    Returns:
        Q in m³/s
    """
    return C * I_mm_hr * A_ha / 360.0


def _manning_pipe_diameter(Q, slope, n=MANNING_N_EARTHEN):
    """Compute required pipe/channel diameter using Manning's equation.

    For circular pipe flowing full:
        Q = (1/n) × A × R^(2/3) × S^(1/2)
        D = (Q × n / (0.3117 × S^0.5))^(3/8)

    Args:
        Q: design discharge (m³/s)
        slope: bed slope (m/m)
        n: Manning's roughness coefficient

    Returns:
        diameter in mm
    """
    if Q <= 0 or slope <= 0:
        return MIN_PIPE_DIAMETER_MM

    S = max(slope, MIN_BED_SLOPE)
    # Manning's for circular pipe: D = (Q*n / (0.3117 * S^0.5))^(3/8)
    D = (Q * n / (0.3117 * S**0.5))**(3.0/8.0)
    D_mm = D * 1000.0

    # Apply freeboard
    D_mm *= (1.0 + FREEBOARD_RATIO)

    # Round up to nearest 50mm standard size
    D_mm = max(MIN_PIPE_DIAMETER_MM, math.ceil(D_mm / 50.0) * 50)
    D_mm = min(D_mm, MAX_PIPE_DIAMETER_MM)

    return int(D_mm)


def _manning_velocity(diameter_mm, slope, n=MANNING_N_EARTHEN):
    """Compute flow velocity using Manning's equation for circular pipe.

    V = (1/n) × R^(2/3) × S^(1/2)
    where R = D/4 for pipe flowing full.
    """
    D = diameter_mm / 1000.0  # metres
    R = D / 4.0               # hydraulic radius
    S = max(slope, MIN_BED_SLOPE)
    V = (1.0 / n) * R**(2.0/3.0) * S**0.5
    return round(V, 3)


def _trench_dimensions(diameter_mm, depth_override=None):
    """Compute trench dimensions for a drain.

    Returns (depth_m, width_m).
    """
    D_m = diameter_mm / 1000.0

    # Depth = pipe diameter + cover (0.3m min) + bedding (0.1m)
    depth = D_m + 0.3 + 0.1
    depth = max(MIN_TRENCH_DEPTH_M, min(depth, MAX_TRENCH_DEPTH_M))

    if depth_override is not None:
        depth = max(MIN_TRENCH_DEPTH_M, min(depth_override, MAX_TRENCH_DEPTH_M))

    # Width at bottom = pipe diameter + 0.3m clearance
    bottom_width = D_m + 0.3

    # Trapezoidal: top width = bottom + 2 × depth × side_slope
    top_width = bottom_width + 2 * depth * SIDE_SLOPE

    return round(depth, 2), round(top_width, 2)


def _compute_bed_slope(path_coords, dtm_arr, gt, pixel_size):
    """Compute bed slope along a drain route from path pixel coordinates.

    Returns slope in m/m (positive = downhill).
    """
    if not path_coords or len(path_coords) < 2:
        return MIN_BED_SLOPE

    # Extract elevations along path
    elevs = []
    for r, c in path_coords:
        if 0 <= r < dtm_arr.shape[0] and 0 <= c < dtm_arr.shape[1]:
            e = float(dtm_arr[r, c])
            if np.isfinite(e) and e != NODATA:
                elevs.append(e)

    if len(elevs) < 2:
        return MIN_BED_SLOPE

    # Compute path length
    total_len = 0.0
    for i in range(1, len(path_coords)):
        dr = path_coords[i][0] - path_coords[i-1][0]
        dc = path_coords[i][1] - path_coords[i-1][1]
        total_len += math.sqrt(dr**2 + dc**2) * pixel_size

    if total_len < 1.0:
        return MIN_BED_SLOPE

    # Overall slope = elevation drop / path length
    elev_drop = elevs[0] - elevs[-1]
    bed_slope = elev_drop / total_len

    # Ensure minimum slope for drainage
    return max(bed_slope, MIN_BED_SLOPE)


# ── Main Entry ─────────────────────────────────────────────────

def run(village_dir, rainfall_intensity=None):
    """Compute drain design parameters for all routed drains.

    Args:
        village_dir: path to village directory
        rainfall_intensity: override rainfall intensity (mm/hr)

    Returns:
        True on success, False on failure
    """
    t_start = time.time()

    terrain_dir = os.path.join(village_dir, "terrain")
    hydro_dir = os.path.join(village_dir, "hydrology")
    wl_dir = os.path.join(village_dir, "waterlogging")
    drain_dir = os.path.join(village_dir, "drainage")

    print("\n" + "=" * 60, flush=True)
    print("  DRAIN DESIGN PARAMETERS (Rational Method)", flush=True)
    print("=" * 60, flush=True)

    # ── Load rasters ──
    dtm_path = os.path.join(terrain_dir, "village_dtm_clean.tif")
    slope_path = os.path.join(terrain_dir, "slope.tif")
    fa_path = os.path.join(hydro_dir, "flow_accumulation.tif")
    risk_path = os.path.join(wl_dir, "waterlogging_risk_class.tif")

    for p in [dtm_path, slope_path, risk_path]:
        if not os.path.isfile(p):
            print(f"  FATAL: Missing {os.path.basename(p)}", flush=True)
            return False

    dtm_arr, gt, proj, dtm_nd = _read_raster(dtm_path)
    slope_arr, _, _, slope_nd = _read_raster(slope_path)
    fa_arr, _, _, fa_nd = _read_raster(fa_path)
    risk_arr, _, _, risk_nd = _read_raster(risk_path)

    rows, cols = dtm_arr.shape
    pixel_size = abs(gt[1])
    pixel_area = pixel_size ** 2
    valid = (dtm_arr != dtm_nd) if dtm_nd is not None else np.ones((rows, cols), bool)

    print(f"  Raster: {rows}x{cols}, pixel={pixel_size:.2f}m", flush=True)

    # ── Terrain classification + runoff coefficient ──
    terrain_type = _classify_terrain(dtm_arr, slope_arr, valid)
    C = RUNOFF_COEFFICIENTS.get(terrain_type, 0.50)
    I = rainfall_intensity or DEFAULT_RAINFALL_INTENSITY
    print(f"  Terrain: {terrain_type}, C={C:.2f}, I={I:.0f} mm/hr", flush=True)

    # ── Load drainage routes from CSV ──
    csv_in = os.path.join(drain_dir, "drainage_design_summary.csv")
    if not os.path.isfile(csv_in):
        print("  SKIP: No drainage_design_summary.csv found", flush=True)
        return True

    routes = []
    with open(csv_in, "r", newline="") as f:
        for row in csv.DictReader(f):
            routes.append(row)

    if not routes:
        print("  SKIP: No drain routes to design", flush=True)
        return True

    print(f"  Routes to design: {len(routes)}", flush=True)

    # ── Identify HIGH clusters for catchment area computation ──
    high_mask = (risk_arr == 2) & valid
    if risk_nd is not None:
        high_mask &= (risk_arr != risk_nd)
    labeled, n_clusters = ndimage_label(high_mask, structure=np.ones((3, 3)))

    # ── Compute design parameters per route ──
    print("\n  Computing design parameters ...", flush=True)
    design_rows = []
    total_Q = 0.0
    total_length = 0.0

    for i, route in enumerate(routes):
        cid = int(route.get("cluster_id", 0))
        area_m2 = float(route.get("area_m2", 0))
        path_len = float(route.get("path_len_m", 0))
        elev_drop = float(route.get("elev_drop_m", 0))
        status = route.get("status", "").strip().upper()

        # Cluster mask
        cluster_mask = (labeled == cid) if cid > 0 else np.zeros((rows, cols), bool)

        # Catchment area (ha) — from flow accumulation at pour point
        if fa_arr is not None and cluster_mask.any():
            A_ha = _compute_catchment_area(
                fa_arr, dtm_arr, valid, cluster_mask, pixel_area, gt)
        else:
            A_ha = area_m2 / 10000.0  # fallback: cluster area

        # Ensure minimum catchment = cluster area
        A_ha = max(A_ha, area_m2 / 10000.0)

        # Bed slope
        if path_len > 0 and elev_drop != 0:
            bed_slope = abs(elev_drop) / path_len
        else:
            bed_slope = MIN_BED_SLOPE
        bed_slope = max(bed_slope, MIN_BED_SLOPE)

        # Rational Method discharge
        Q = _rational_method(C, I, A_ha)

        # Pipe sizing (Manning's)
        diameter_mm = _manning_pipe_diameter(Q, bed_slope)

        # Flow velocity
        velocity = _manning_velocity(diameter_mm, bed_slope)

        # Trench dimensions
        trench_depth, trench_width = _trench_dimensions(diameter_mm)

        # Feasibility assessment
        if status in ("ROUTED", "DIRECT_OUTLET") and elev_drop > 0:
            feasibility = "GRAVITY"
        elif status in ("ROUTED", "DIRECT_OUTLET"):
            feasibility = "PUMPED"
        else:
            feasibility = "UNROUTABLE"

        design = {
            "cluster_id": cid,
            "area_m2": round(area_m2, 1),
            "catchment_area_ha": round(A_ha, 3),
            "runoff_coefficient": C,
            "rainfall_intensity_mm_hr": I,
            "peak_discharge_m3s": round(Q, 4),
            "pipe_diameter_mm": diameter_mm,
            "bed_slope_m_m": round(bed_slope, 5),
            "manning_velocity_ms": velocity,
            "trench_depth_m": trench_depth,
            "trench_width_m": trench_width,
            "path_len_m": round(path_len, 1),
            "elev_drop_m": round(elev_drop, 2),
            "status": status,
            "feasibility": feasibility,
        }
        design_rows.append(design)

        total_Q += Q
        if status in ("ROUTED", "DIRECT_OUTLET"):
            total_length += path_len

        if (i + 1) % 50 == 0 or (i + 1) == len(routes):
            print(f"    [{i+1}/{len(routes)}] designed", flush=True)

    # ── Write design CSV ──
    csv_out = os.path.join(drain_dir, "drain_design_parameters.csv")
    fieldnames = [
        "cluster_id", "area_m2", "catchment_area_ha",
        "runoff_coefficient", "rainfall_intensity_mm_hr",
        "peak_discharge_m3s", "pipe_diameter_mm", "bed_slope_m_m",
        "manning_velocity_ms", "trench_depth_m", "trench_width_m",
        "path_len_m", "elev_drop_m", "status", "feasibility",
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(design_rows)
    print(f"\n  Wrote: {os.path.basename(csv_out)} ({len(design_rows)} rows)",
          flush=True)

    # ── Aggregate summary ──
    n_gravity = sum(1 for d in design_rows if d["feasibility"] == "GRAVITY")
    n_pumped = sum(1 for d in design_rows if d["feasibility"] == "PUMPED")
    n_unroutable = sum(1 for d in design_rows
                       if d["feasibility"] == "UNROUTABLE")
    diameters = [d["pipe_diameter_mm"] for d in design_rows
                 if d["feasibility"] != "UNROUTABLE"]
    discharges = [d["peak_discharge_m3s"] for d in design_rows
                  if d["feasibility"] != "UNROUTABLE"]

    summary = {
        "terrain_type": terrain_type,
        "runoff_coefficient": C,
        "rainfall_intensity_mm_hr": I,
        "n_drains_total": len(design_rows),
        "n_gravity_feasible": n_gravity,
        "n_pumped_required": n_pumped,
        "n_unroutable": n_unroutable,
        "total_drain_length_km": round(total_length / 1000.0, 2),
        "total_peak_discharge_m3s": round(total_Q, 3),
        "mean_pipe_diameter_mm": (
            round(float(np.mean(diameters)), 0) if diameters else 0),
        "max_pipe_diameter_mm": max(diameters) if diameters else 0,
        "min_pipe_diameter_mm": min(diameters) if diameters else 0,
        "mean_discharge_m3s": (
            round(float(np.mean(discharges)), 4) if discharges else 0),
        "max_discharge_m3s": (
            round(float(max(discharges)), 4) if discharges else 0),
    }

    json_out = os.path.join(drain_dir, "drain_design_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote: {os.path.basename(json_out)}", flush=True)

    # ── Print summary ──
    elapsed = time.time() - t_start
    print(f"\n  ── Design Summary ──", flush=True)
    print(f"  Terrain: {terrain_type}, C={C:.2f}, I={I:.0f} mm/hr", flush=True)
    print(f"  Drains: {n_gravity} gravity + {n_pumped} pumped + "
          f"{n_unroutable} unroutable = {len(design_rows)} total", flush=True)
    print(f"  Total drain: {total_length/1000:.2f} km", flush=True)
    print(f"  Peak Q: {total_Q:.3f} m³/s  "
          f"(mean {summary['mean_discharge_m3s']:.4f} m³/s)", flush=True)
    print(f"  Pipe diameters: {summary['min_pipe_diameter_mm']}-"
          f"{summary['max_pipe_diameter_mm']} mm "
          f"(mean {summary['mean_pipe_diameter_mm']:.0f} mm)", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)

    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--village-dir", required=True)
    p.add_argument("--rainfall", type=float, default=None,
                   help="Override rainfall intensity (mm/hr)")
    args = p.parse_args()
    ok = run(args.village_dir, rainfall_intensity=args.rainfall)
    sys.exit(0 if ok else 1)
