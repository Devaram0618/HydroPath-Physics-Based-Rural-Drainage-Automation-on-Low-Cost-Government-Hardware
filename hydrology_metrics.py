"""
engine/hydrology_metrics.py — Hydrology Feature Computation
=============================================================
Computes derived hydrology rasters from terrain data:
  1. TWI (Topographic Wetness Index)
  2. Distance-to-stream
  3. Relative elevation (DEM - local mean)
  4. Convergence index (WhiteboxTools)
  5. Flow accumulation (WhiteboxTools D8)

Inputs  (from <village_dir>/terrain/):
  village_dtm_clean.tif, slope.tif, filled_dtm.tif

Outputs (to <village_dir>/hydrology/):
  flow_accumulation.tif, twi.tif, dist_to_stream.tif,
  relative_elevation.tif, convergence_index.tif
"""
import os
import sys
import subprocess
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from osgeo import gdal

gdal.UseExceptions()

# ── Parameters ─────────────────────────────────────────────────
STREAM_THRESHOLD = 1000    # flow accumulation cells for stream definition
REL_ELEV_WINDOW = 51       # 51px ≈ 50 m at 1 m resolution
NODATA = -9999.0

# WhiteboxTools path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WBT_EXE = os.path.join(PROJECT_DIR, "whitebox",
                        "WhiteboxTools_win_amd64", "WBT",
                        "whitebox_tools.exe")


def _read_raster(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return arr, gt, proj, nd


def _write_raster(path, arr, gt, proj, nodata=NODATA):
    rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    tmp = path + ".writing.tif"
    ds = driver.Create(tmp, cols, rows, 1, gdal.GDT_Float32,
                       options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr.astype(np.float32))
    band.FlushCache()
    ds = None
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rename(tmp, path)
    except OSError:
        path = tmp
    print(f"  Wrote: {os.path.basename(path)} "
          f"({os.path.getsize(path)/1024/1024:.1f} MB)", flush=True)


def _run_cmd(cmd, desc):
    kwargs = dict(capture_output=True, text=True)
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    if isinstance(cmd, str):
        kwargs["shell"] = True
    r = subprocess.run(cmd, **kwargs)
    if r.returncode != 0:
        print(f"  ERROR in {desc}: {r.stderr[:500]}", flush=True)
        return False
    return True


def _banner(step, title):
    print(f"\n{'='*60}", flush=True)
    print(f"STEP {step} — {title}", flush=True)
    print(f"{'='*60}", flush=True)


def _compute_convergence_proxy(dem_arr, valid_mask):
    """Compute a lightweight convergence proxy from DEM gradients.

    Positive values indicate locally convergent downslope flow; negative values
    indicate divergent areas. Output is scaled for downstream model compatibility.
    """
    dem = dem_arr.astype(np.float64, copy=True)
    fill_val = np.median(dem[valid_mask])
    dem[~valid_mask] = fill_val

    gy, gx = np.gradient(dem)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-9
    ux = -gx / mag
    uy = -gy / mag

    # divergence of downslope unit vector field
    div = np.gradient(ux, axis=1) + np.gradient(uy, axis=0)
    conv = -div * 100.0
    out = np.full(dem.shape, NODATA, dtype=np.float64)
    out[valid_mask] = conv[valid_mask]
    return out


def run(village_dir):
    """Compute all hydrology metrics for a village.

    Reads from village_dir/terrain/, writes to village_dir/hydrology/.
    """
    t_total = time.time()

    terrain_dir = os.path.join(village_dir, "terrain")
    hydro_dir = os.path.join(village_dir, "hydrology")
    os.makedirs(hydro_dir, exist_ok=True)

    # Input paths
    dtm_path = os.path.join(terrain_dir, "village_dtm_clean.tif")
    slope_path = os.path.join(terrain_dir, "slope.tif")
    filled_path = os.path.join(terrain_dir, "filled_dtm.tif")

    # Output paths
    flowacc_path = os.path.join(hydro_dir, "flow_accumulation.tif")
    twi_path = os.path.join(hydro_dir, "twi.tif")
    dist_path = os.path.join(hydro_dir, "dist_to_stream.tif")
    rel_path = os.path.join(hydro_dir, "relative_elevation.tif")
    conv_path = os.path.join(hydro_dir, "convergence_index.tif")

    print("=" * 60, flush=True)
    print("  HYDROLOGY METRICS", flush=True)
    print("=" * 60, flush=True)

    for p in [dtm_path, slope_path, filled_path]:
        if not os.path.exists(p):
            print(f"  FATAL: Missing {p}", flush=True)
            raise RuntimeError(f"Missing required raster: {p}")

    # Read reference rasters
    dtm_arr, gt, proj, dtm_nd = _read_raster(dtm_path)
    slope_arr, _, _, slope_nd = _read_raster(slope_path)
    valid = dtm_arr != dtm_nd
    rows, cols = dtm_arr.shape
    slope_valid = slope_arr != slope_nd if slope_nd is not None else np.ones_like(slope_arr, dtype=bool)

    print(f"  Raster: {rows}×{cols}, {valid.sum():,} valid", flush=True)

    # ── STEP 1: Flow accumulation (WhiteboxTools D8) ──
    _banner(1, "D8 Flow Accumulation")
    t0 = time.time()

    if os.path.exists(flowacc_path) and os.path.getsize(flowacc_path) > 1000:
        print("  SKIPPED (cached)", flush=True)
    else:
        _run_cmd(
            f'"{WBT_EXE}" -r=D8FlowAccumulation '
            f'-i="{filled_path}" -o="{flowacc_path}" '
            f'--out_type=cells -v',
            "D8FlowAccumulation")
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    flowacc_arr, _, _, fa_nd = _read_raster(flowacc_path)
    fa_valid = (flowacc_arr != fa_nd) if fa_nd is not None else np.ones_like(flowacc_arr, dtype=bool)
    tri_valid = valid & slope_valid & fa_valid

    # ── STEP 2: TWI ──
    _banner(2, "Topographic Wetness Index")
    t0 = time.time()

    slope_rad = np.deg2rad(np.clip(slope_arr, 0, 89.9))
    eps = 1e-6
    twi = np.full((rows, cols), NODATA, dtype=np.float64)
    twi[tri_valid] = np.log(
        (flowacc_arr[tri_valid] + 1.0) /
        (np.tan(slope_rad[tri_valid]) + eps))
    _write_raster(twi_path, twi, gt, proj)
    v = twi[tri_valid]
    print(f"  TWI range: {v.min():.2f}..{v.max():.2f} (mean={v.mean():.2f})",
          flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ── STEP 3: Distance to stream ──
    _banner(3, "Distance to Stream")
    t0 = time.time()

    stream_mask = (flowacc_arr > STREAM_THRESHOLD) & fa_valid
    n_stream = stream_mask.sum()
    print(f"  Stream cells (>{STREAM_THRESHOLD}): {n_stream:,}", flush=True)

    if n_stream == 0:
        alt = 500
        stream_mask = (flowacc_arr > alt) & fa_valid
        n_stream = stream_mask.sum()
        print(f"  Lowered to >{alt}: {n_stream:,}", flush=True)

    edt_in = (~stream_mask).astype(np.float64)
    dist_raw = distance_transform_edt(edt_in)
    dist = np.full((rows, cols), NODATA, dtype=np.float64)
    dist[tri_valid] = dist_raw[tri_valid]
    _write_raster(dist_path, dist, gt, proj)
    d = dist[tri_valid]
    print(f"  Dist range: {d.min():.1f}..{d.max():.1f} m", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ── STEP 4: Relative elevation ──
    _banner(4, "Relative Elevation")
    t0 = time.time()

    med = np.median(dtm_arr[valid])
    filled = dtm_arr.copy()
    filled[~valid] = med
    local_mean = uniform_filter(filled, size=REL_ELEV_WINDOW)
    rel = np.full((rows, cols), NODATA, dtype=np.float64)
    rel[valid] = dtm_arr[valid] - local_mean[valid]
    _write_raster(rel_path, rel, gt, proj)
    r = rel[valid]
    print(f"  Range: {r.min():.2f}..{r.max():.2f} m (mean={r.mean():.2f})",
          flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ── STEP 5: Convergence index ──
    _banner(5, "Convergence Index (WhiteboxTools)")
    t0 = time.time()

    if os.path.exists(conv_path) and os.path.getsize(conv_path) > 1000:
        print("  SKIPPED (cached)", flush=True)
    else:
        ok = _run_cmd(
            f'"{WBT_EXE}" -r=ConvergenceIndex '
            f'--dem="{filled_path}" --output="{conv_path}" -v',
            "ConvergenceIndex")
        if not ok or (not os.path.exists(conv_path)) or os.path.getsize(conv_path) < 1000:
            print("  Falling back to DEM-gradient convergence proxy", flush=True)
            conv_proxy = _compute_convergence_proxy(dtm_arr, valid)
            _write_raster(conv_path, conv_proxy, gt, proj)

    conv_arr, _, _, conv_nd = _read_raster(conv_path)
    conv_valid = (conv_arr != conv_nd) if conv_nd is not None else np.ones_like(conv_arr, dtype=bool)
    both = valid & conv_valid
    conv_final = np.full((rows, cols), NODATA, dtype=np.float64)
    conv_final[both] = conv_arr[both]
    _write_raster(conv_path, conv_final, gt, proj)
    c = conv_final[both]
    print(f"  Range: {c.min():.2f}..{c.max():.2f} (mean={c.mean():.2f})",
          flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t_total
    print(f"\n  Total hydrology time: {elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--village-dir", required=True)
    args = p.parse_args()
    run(args.village_dir)
