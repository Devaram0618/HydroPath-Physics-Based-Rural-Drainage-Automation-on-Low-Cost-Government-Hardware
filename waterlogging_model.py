"""
engine/waterlogging_model.py — Deterministic Physics-Based Waterlogging
========================================================================
Replaces GMM clustering with a physically constrained scoring model.

Algorithm:
  STEP 1 — Robust-scale 5 features (TWI, relative_elevation,
            convergence_index, slope, distance_to_stream).
  STEP 2 — Basin eligibility mask:
            relative_elevation < -0.25 m
            AND convergence_index > 0
            AND slope < 5 degrees.
  STEP 3 — Weighted physics score (deterministic).
  STEP 4 — Class assignment: HIGH = top 15% eligible, MEDIUM = next 25%.
  STEP 5 — HPF refinement: remove HIGH where slope > 6 OR rel_elev > -0.1.

Outputs (written to <village_dir>/waterlogging/):
  waterlogging_score.tif       — continuous score (0..1)
  waterlogging_risk_class.tif  — 0=LOW, 1=MEDIUM, 2=HIGH

No randomness. No clustering. Pure physics-aligned scoring.
"""
import os
import sys
import time
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# ── THRESHOLD CONSTANTS (defined at top, per spec) ─────────────
# Basin eligibility (STEP 2)
BASIN_REL_ELEV_MAX = -0.25      # metres — must be below local mean
BASIN_CONVERGENCE_MIN = 0.0     # positive convergence required
BASIN_SLOPE_MAX = 5.0           # degrees — flat enough to pond

# Score weights (STEP 3)
W_TWI = +0.35
W_CONVERGENCE = +0.25
W_REL_ELEV = -0.25             # negative: lower rel_elev → higher score
W_SLOPE = -0.10                # negative: lower slope → higher score
W_DIST_STREAM = -0.05          # negative: closer to stream → higher score

# Class thresholds (STEP 4)
HIGH_PERCENTILE = 85           # top 15% of eligible → HIGH
MEDIUM_PERCENTILE = 60         # next 25% → MEDIUM

# HPF thresholds (STEP 5)
HPF_SLOPE_MAX = 6.0            # degrees — slope above this cannot be HIGH
HPF_REL_ELEV_MAX = -0.1        # metres — above this cannot be HIGH

NODATA = -9999.0


# ── RASTER I/O ─────────────────────────────────────────────────

def _read_raster(path):
    """Read single-band raster → (array_f64, geotransform, projection, nodata)."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return arr, gt, proj, nd


def _write_raster(path, arr, gt, proj, nodata=NODATA, dtype=gdal.GDT_Float32):
    """Write numpy array → GeoTIFF with safe overwrite."""
    rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    tmp_path = path + ".writing.tif"
    ds = driver.Create(tmp_path, cols, rows, 1, dtype,
                       options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr.astype(np.float32))
    band.FlushCache()
    ds = None
    # Atomic replace
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rename(tmp_path, path)
    except OSError:
        # Fallback: keep temp
        print(f"  WARNING: Could not replace {path}, kept {tmp_path}",
              flush=True)
        path = tmp_path
    print(f"  Wrote: {os.path.basename(path)} "
          f"({os.path.getsize(path)/1024/1024:.1f} MB)", flush=True)
    return path


def _banner(step, title):
    print(f"\n{'='*60}", flush=True)
    print(f"STEP {step} — {title}", flush=True)
    print(f"{'='*60}", flush=True)


# ── ROBUST SCALING ─────────────────────────────────────────────

def _robust_scale(values):
    """Robust-scale a 1-D array: (x - median) / IQR.

    Returns scaled array. If IQR == 0, uses std instead.
    """
    med = np.median(values)
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25
    if iqr < 1e-10:
        iqr = max(np.std(values), 1e-10)
    return (values - med) / iqr


# ── MAIN ENTRY POINT ──────────────────────────────────────────

def run(village_dir):
    """Run deterministic waterlogging classification for one village.

    Parameters
    ----------
    village_dir : str
        Path to village directory (new structure) containing:
          terrain/slope.tif
          hydrology/twi.tif
          hydrology/relative_elevation.tif
          hydrology/convergence_index.tif
          hydrology/dist_to_stream.tif

    Outputs
    -------
    Writes to village_dir/waterlogging/:
      waterlogging_score.tif
      waterlogging_risk_class.tif

    Returns
    -------
    dict with summary metrics.
    """
    t_total = time.time()

    terrain_dir = os.path.join(village_dir, "terrain")
    hydro_dir = os.path.join(village_dir, "hydrology")
    out_dir = os.path.join(village_dir, "waterlogging")
    os.makedirs(out_dir, exist_ok=True)

    score_tif = os.path.join(out_dir, "waterlogging_score.tif")
    risk_tif = os.path.join(out_dir, "waterlogging_risk_class.tif")

    print("=" * 60, flush=True)
    print("  WATERLOGGING MODEL v2 — Physics-Based Scoring", flush=True)
    print("=" * 60, flush=True)
    print(f"  Village: {os.path.basename(village_dir)}", flush=True)
    print(f"  Mode: DETERMINISTIC (no clustering, no randomness)", flush=True)

    # ── Load input rasters ──────────────────────────────────────
    _banner("0", "Load Input Rasters")

    paths = {
        "slope":       os.path.join(terrain_dir, "slope.tif"),
        "twi":         os.path.join(hydro_dir, "twi.tif"),
        "rel_elev":    os.path.join(hydro_dir, "relative_elevation.tif"),
        "convergence": os.path.join(hydro_dir, "convergence_index.tif"),
        "dist_stream": os.path.join(hydro_dir, "dist_to_stream.tif"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"  FATAL: Missing {name}: {path}", flush=True)
            raise RuntimeError(f"Missing {name}: {path}")
        print(f"  {name:18s} → {os.path.basename(path)}", flush=True)

    # Read all rasters
    slope_arr, gt, proj, slope_nd = _read_raster(paths["slope"])
    twi_arr, _, _, twi_nd = _read_raster(paths["twi"])
    rel_arr, _, _, rel_nd = _read_raster(paths["rel_elev"])
    conv_arr, _, _, conv_nd = _read_raster(paths["convergence"])
    dist_arr, _, _, dist_nd = _read_raster(paths["dist_stream"])

    rows, cols = slope_arr.shape
    print(f"\n  Raster: {rows} × {cols} ({rows*cols:,} cells)", flush=True)

    # Build combined valid mask (all 5 features present)
    valid = np.ones((rows, cols), dtype=bool)
    for arr, nd in [(slope_arr, slope_nd), (twi_arr, twi_nd),
                    (rel_arr, rel_nd), (conv_arr, conv_nd),
                    (dist_arr, dist_nd)]:
        if nd is not None:
            valid &= (arr != nd)

    n_valid = valid.sum()
    print(f"  Valid cells: {n_valid:,} ({100*n_valid/(rows*cols):.1f}%)",
          flush=True)

    if n_valid == 0:
        print("  FATAL: No valid cells — aborting.", flush=True)
        raise RuntimeError("No valid cells in rasters")

    # ================================================================
    # STEP 1 — Robust-scale features
    # ================================================================
    _banner(1, "Robust Scaling (5 features)")
    t0 = time.time()

    feat_names = ["twi", "convergence", "rel_elev", "slope", "dist_stream"]
    raw_arrays = {
        "twi": twi_arr,
        "convergence": conv_arr,
        "rel_elev": rel_arr,
        "slope": slope_arr,
        "dist_stream": dist_arr,
    }

    scaled = {}
    for name in feat_names:
        vals = raw_arrays[name][valid]
        s = _robust_scale(vals)
        # Store back into a full raster (nodata = 0 for scoring)
        full = np.zeros((rows, cols), dtype=np.float32)
        full[valid] = s.astype(np.float32)
        scaled[name] = full
        print(f"  {name:18s}  median_scaled={np.median(s):+.3f}  "
              f"IQR=[{np.percentile(s,25):+.3f}, {np.percentile(s,75):+.3f}]",
              flush=True)

    # Free raw arrays to reclaim memory
    del raw_arrays
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # STEP 2 — Basin eligibility mask
    # ================================================================
    _banner(2, "Basin Eligibility Mask")
    t0 = time.time()

    eligible = (
        valid &
        (rel_arr < BASIN_REL_ELEV_MAX) &
        (conv_arr > BASIN_CONVERGENCE_MIN) &
        (slope_arr < BASIN_SLOPE_MAX)
    )

    n_eligible = eligible.sum()
    pct_eligible = 100 * n_eligible / max(n_valid, 1)
    print(f"  Criteria:", flush=True)
    print(f"    relative_elevation < {BASIN_REL_ELEV_MAX} m", flush=True)
    print(f"    convergence_index  > {BASIN_CONVERGENCE_MIN}", flush=True)
    print(f"    slope              < {BASIN_SLOPE_MAX}°", flush=True)
    print(f"  Eligible pixels: {n_eligible:,} ({pct_eligible:.1f}% of valid)",
          flush=True)

    if n_eligible == 0:
        print("  WARNING: Zero eligible pixels! Relaxing criteria ...",
              flush=True)
        # Fallback: use only rel_elev and slope constraints
        eligible = (
            valid &
            (rel_arr < 0.0) &
            (slope_arr < BASIN_SLOPE_MAX * 2)
        )
        n_eligible = eligible.sum()
        pct_eligible = 100 * n_eligible / max(n_valid, 1)
        print(f"  Relaxed eligible: {n_eligible:,} ({pct_eligible:.1f}%)",
              flush=True)

    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # STEP 3 — Physics risk score
    # ================================================================
    _banner(3, "Physics Risk Score")
    t0 = time.time()

    score = np.full((rows, cols), NODATA, dtype=np.float32)

    # Compute score for ALL valid pixels incrementally (memory-safe)
    score[valid] = 0.0
    for feat, w in [("twi", W_TWI), ("convergence", W_CONVERGENCE),
                    ("rel_elev", W_REL_ELEV), ("slope", W_SLOPE),
                    ("dist_stream", W_DIST_STREAM)]:
        score[valid] += np.float32(w) * scaled[feat][valid].astype(np.float32)

    # Free scaled features to reclaim memory
    del scaled
    import gc; gc.collect()

    score_vals = score[valid]
    print(f"  Score range: {score_vals.min():.4f} to {score_vals.max():.4f}",
          flush=True)
    print(f"  Score mean:  {score_vals.mean():.4f}", flush=True)
    print(f"  Score std:   {score_vals.std():.4f}", flush=True)

    # Normalize score to [0, 1] for the output raster
    s_min, s_max = float(score_vals.min()), float(score_vals.max())
    s_range = s_max - s_min if (s_max - s_min) > 1e-10 else 1.0
    score_norm = np.full((rows, cols), NODATA, dtype=np.float32)
    score_norm[valid] = (score[valid] - s_min) / s_range

    _write_raster(score_tif, score_norm, gt, proj)

    print(f"  Weights: TWI={W_TWI:+.2f}  conv={W_CONVERGENCE:+.2f}  "
          f"rel_el={W_REL_ELEV:+.2f}  slope={W_SLOPE:+.2f}  "
          f"dist={W_DIST_STREAM:+.2f}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # STEP 4 — Class assignment
    # ================================================================
    _banner(4, "Class Assignment (percentile-based)")
    t0 = time.time()

    # Extract scores for ELIGIBLE pixels only
    eligible_scores = score[eligible]

    if len(eligible_scores) > 0:
        high_thresh = np.percentile(eligible_scores, HIGH_PERCENTILE)
        med_thresh = np.percentile(eligible_scores, MEDIUM_PERCENTILE)
    else:
        high_thresh = np.inf
        med_thresh = np.inf

    print(f"  HIGH threshold:   score > {high_thresh:.4f} "
          f"(top {100-HIGH_PERCENTILE}% of eligible)", flush=True)
    print(f"  MEDIUM threshold: score > {med_thresh:.4f} "
          f"(next {HIGH_PERCENTILE-MEDIUM_PERCENTILE}%)", flush=True)

    # Assign classes: default LOW
    risk_class = np.full((rows, cols), NODATA, dtype=np.float32)
    risk_class[valid] = 0.0  # LOW

    # MEDIUM: eligible pixels above medium threshold
    medium_mask = eligible & (score >= med_thresh) & (score < high_thresh)
    risk_class[medium_mask] = 1.0

    # HIGH: eligible pixels above high threshold
    high_mask = eligible & (score >= high_thresh)
    risk_class[high_mask] = 2.0

    n_low = int((risk_class[valid] == 0).sum())
    n_med = int((risk_class[valid] == 1).sum())
    n_high = int((risk_class[valid] == 2).sum())

    print(f"\n  Pre-HPF distribution:", flush=True)
    print(f"    LOW:    {n_low:>8,} ({100*n_low/n_valid:.1f}%)", flush=True)
    print(f"    MEDIUM: {n_med:>8,} ({100*n_med/n_valid:.1f}%)", flush=True)
    print(f"    HIGH:   {n_high:>8,} ({100*n_high/n_valid:.1f}%)", flush=True)

    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # STEP 5 — HPF (Hydrological Plausibility Filter)
    # ================================================================
    _banner(5, "HPF — Refined Plausibility Filter")
    t0 = time.time()

    pre_high = (risk_class == 2)
    n_pre = int(pre_high.sum())

    # Remove HIGH where slope > 6° OR relative_elevation > -0.1m
    impossible = pre_high & (
        (slope_arr > HPF_SLOPE_MAX) |
        (rel_arr > HPF_REL_ELEV_MAX)
    )
    n_demoted = int(impossible.sum())
    risk_class[impossible] = 1.0  # demote to MEDIUM

    pct_demoted = 100 * n_demoted / max(n_pre, 1)
    print(f"  HPF criteria:", flush=True)
    print(f"    slope > {HPF_SLOPE_MAX}°  OR  rel_elev > {HPF_REL_ELEV_MAX} m",
          flush=True)
    print(f"  Demoted: {n_demoted:,} HIGH → MEDIUM "
          f"({pct_demoted:.1f}% of {n_pre:,})", flush=True)

    # Final counts
    n_low = int((risk_class[valid] == 0).sum())
    n_med = int((risk_class[valid] == 1).sum())
    n_high = int((risk_class[valid] == 2).sum())
    pct_high = 100 * n_high / n_valid

    print(f"\n  Final distribution:", flush=True)
    print(f"    LOW:    {n_low:>8,} ({100*n_low/n_valid:.1f}%)", flush=True)
    print(f"    MEDIUM: {n_med:>8,} ({100*n_med/n_valid:.1f}%)", flush=True)
    print(f"    HIGH:   {n_high:>8,} ({pct_high:.1f}%)", flush=True)

    _write_raster(risk_tif, risk_class, gt, proj)

    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # Summary + sanity checks
    # ================================================================
    _banner("✓", "Validation Summary")

    metrics = {
        "n_valid": int(n_valid),
        "n_eligible": int(n_eligible),
        "pct_eligible": round(pct_eligible, 2),
        "n_high": n_high,
        "n_medium": n_med,
        "n_low": n_low,
        "pct_high": round(pct_high, 2),
        "hpf_demoted": n_demoted,
    }

    if n_high > 0:
        high_final = (risk_class == 2)
        mean_rel = float(rel_arr[high_final & valid].mean())
        mean_slope = float(slope_arr[high_final & valid].mean())
        mean_twi = float(twi_arr[high_final & valid].mean())
        mean_conv = float(conv_arr[high_final & valid].mean())

        print(f"\n  HIGH-zone physics check:", flush=True)
        print(f"    Mean relative_elevation: {mean_rel:+.3f} m "
              f"(should be < 0)", flush=True)
        print(f"    Mean slope:              {mean_slope:.2f}° "
              f"(should be < {HPF_SLOPE_MAX}°)", flush=True)
        print(f"    Mean TWI:                {mean_twi:.2f}", flush=True)
        print(f"    Mean convergence:        {mean_conv:+.2f}", flush=True)

        metrics["high_mean_rel_elev"] = round(mean_rel, 3)
        metrics["high_mean_slope"] = round(mean_slope, 2)
        metrics["high_mean_twi"] = round(mean_twi, 2)
        metrics["high_mean_convergence"] = round(mean_conv, 2)

        # Check overlap with top 30% flow accumulation
        flowacc_path = os.path.join(hydro_dir, "flow_accumulation.tif")
        if os.path.exists(flowacc_path):
            fa_arr, _, _, fa_nd = _read_raster(flowacc_path)
            fa_valid = valid & (fa_arr != fa_nd) if fa_nd is not None else valid
            if fa_valid.any():
                fa_p70 = np.percentile(fa_arr[fa_valid], 70)
                top30_fa = fa_valid & (fa_arr >= fa_p70)
                overlap = (high_final & top30_fa).sum()
                pct_overlap = 100 * overlap / max(n_high, 1)
                print(f"    HIGH ∩ top-30% flow_acc: {pct_overlap:.1f}%",
                      flush=True)
                metrics["high_flowacc_overlap_pct"] = round(pct_overlap, 1)

    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)
    print("  WATERLOGGING MODEL v2 — COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)

    metrics["elapsed_s"] = round(elapsed, 1)
    return metrics


# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Physics-based waterlogging classification")
    parser.add_argument("--village-dir", required=True,
                        help="Path to village directory (new structure)")
    args = parser.parse_args()
    metrics = run(args.village_dir)
    print(f"\nMetrics: {metrics}")
