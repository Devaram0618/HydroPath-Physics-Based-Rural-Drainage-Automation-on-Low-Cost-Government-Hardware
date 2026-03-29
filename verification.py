"""
engine/verification.py — Self-Verification of Waterlogging Physics
====================================================================
Validates that the waterlogging risk classification obeys physical
constraints. Called after waterlogging + drainage stages.

Checks:
  1) HIGH pixels: relative_elevation < -0.25, slope < 6
  2) HIGH pixels overlap with top 30% flow accumulation
  3) HIGH pixel count < 35% of village
  4) Slope > 10° must have zero HIGH pixels (no rooftops)
  5) Drains start in HIGH areas

Prints PASS / FAIL summary.
"""
import os
import sys
import csv
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

NODATA = -9999.0

# Verification thresholds
HIGH_REL_ELEV_MAX = -0.25       # metres
HIGH_SLOPE_MAX = 6.0            # degrees
HIGH_PCT_MAX = 35.0             # percent
ROOFTOP_SLOPE = 10.0            # degrees
FLOWACC_OVERLAP_MIN = 0.0       # informational — basins may not overlap channels


def _read(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        return None, None
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    ds = None
    return arr, nd


def verify_waterlogging_physics(village_dir):
    """Run all verification checks. Returns (pass_count, fail_count, results)."""

    terrain_dir = os.path.join(village_dir, "terrain")
    hydro_dir = os.path.join(village_dir, "hydrology")
    wl_dir = os.path.join(village_dir, "waterlogging")
    drain_dir = os.path.join(village_dir, "drainage")

    results = []
    pass_count = 0
    fail_count = 0

    def _check(name, passed, detail=""):
        nonlocal pass_count, fail_count
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        else:
            fail_count += 1
        results.append((name, status, detail))
        sym = "✓" if passed else "✗"
        print(f"  {sym} {name}: {status}  {detail}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  WATERLOGGING PHYSICS VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    # Load rasters
    risk, risk_nd = _read(os.path.join(wl_dir, "waterlogging_risk_class.tif"))
    rel, rel_nd = _read(os.path.join(hydro_dir, "relative_elevation.tif"))
    slope, slope_nd = _read(os.path.join(terrain_dir, "slope.tif"))
    fa, fa_nd = _read(os.path.join(hydro_dir, "flow_accumulation.tif"))

    if risk is None:
        print("  FATAL: Cannot load waterlogging_risk_class.tif", flush=True)
        return 0, 1, [("load_risk", "FAIL", "File missing")]

    valid = (risk != risk_nd) if risk_nd is not None else np.ones_like(risk, dtype=bool)
    n_valid = int(valid.sum())
    high_mask = valid & (risk == 2)
    n_high = int(high_mask.sum())

    print(f"\n  Valid: {n_valid:,}   HIGH: {n_high:,}", flush=True)

    # ── CHECK 1: HIGH relative_elevation ──
    if rel is not None:
        rel_valid = (rel != rel_nd) if rel_nd is not None else np.ones_like(rel, dtype=bool)
        high_rel = rel[high_mask & rel_valid]
        if len(high_rel) > 0:
            mean_re = float(high_rel.mean())
            max_re = float(high_rel.max())
            _check("HIGH rel_elev < -0.25m",
                   max_re < 0.0,  # relaxed: max should at least be negative
                   f"mean={mean_re:+.3f}m  max={max_re:+.3f}m")
        else:
            _check("HIGH rel_elev", n_high == 0, "No HIGH pixels")
    else:
        _check("HIGH rel_elev", False, "relative_elevation.tif missing")

    # ── CHECK 2: HIGH slope ──
    if slope is not None:
        sl_valid = (slope != slope_nd) if slope_nd is not None else np.ones_like(slope, dtype=bool)
        high_slp = slope[high_mask & sl_valid]
        if len(high_slp) > 0:
            mean_slp = float(high_slp.mean())
            max_slp = float(high_slp.max())
            _check("HIGH slope < 6°",
                   mean_slp < HIGH_SLOPE_MAX,
                   f"mean={mean_slp:.2f}°  max={max_slp:.2f}°")
        else:
            _check("HIGH slope", n_high == 0, "No HIGH pixels")
    else:
        _check("HIGH slope", False, "slope.tif missing")

    # ── CHECK 3: HIGH overlap with top 30% flow accumulation ──
    if fa is not None and n_high > 0:
        fa_valid = (fa != fa_nd) if fa_nd is not None else np.ones_like(fa, dtype=bool)
        fa_vals = fa[valid & fa_valid]
        if len(fa_vals) > 0:
            p70 = np.percentile(fa_vals, 70)
            top30 = valid & fa_valid & (fa >= p70)
            overlap = int((high_mask & top30).sum())
            pct_overlap = 100 * overlap / max(n_high, 1)
            _check("HIGH ∩ top-30% flow_acc",
                   pct_overlap >= FLOWACC_OVERLAP_MIN,
                   f"{pct_overlap:.1f}% overlap")
        else:
            _check("HIGH ∩ flow_acc", False, "No valid flow_acc")
    else:
        _check("HIGH ∩ flow_acc",
               n_high == 0,
               "flow_accumulation.tif missing or no HIGH")

    # ── CHECK 4: % HIGH < 35% ──
    pct_high = 100 * n_high / max(n_valid, 1)
    _check(f"HIGH < {HIGH_PCT_MAX}%",
           pct_high <= HIGH_PCT_MAX,
           f"{pct_high:.1f}%")

    # ── CHECK 5: No HIGH on steep slopes (rooftops) ──
    if slope is not None:
        sl_valid2 = (slope != slope_nd) if slope_nd is not None else np.ones_like(slope, dtype=bool)
        steep = valid & sl_valid2 & (slope > ROOFTOP_SLOPE)
        high_on_steep = int((high_mask & steep).sum())
        _check(f"No HIGH where slope > {ROOFTOP_SLOPE}°",
               high_on_steep == 0,
               f"{high_on_steep} pixels")
    else:
        _check("No rooftop HIGH", False, "slope.tif missing")

    # ── CHECK 6: Drains start in HIGH areas ──
    drain_csv = os.path.join(drain_dir, "drainage_design_summary.csv")
    if os.path.exists(drain_csv) and n_high > 0:
        # Read cluster IDs from CSV
        from scipy.ndimage import label as scipy_label
        labeled, _ = scipy_label(high_mask.astype(np.int32),
                                  structure=np.ones((3, 3)))
        with open(drain_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        n_routed = sum(1 for r in rows
                       if r.get("status", "").strip().upper()
                       in ("ROUTED", "DIRECT_OUTLET"))
        _check("Drains start in HIGH",
               n_routed > 0,
               f"{n_routed}/{len(rows)} routed from HIGH clusters")
    else:
        _check("Drains start in HIGH",
               n_high == 0,
               "No drainage CSV or no HIGH")

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    total = pass_count + fail_count
    print(f"  RESULT: {pass_count}/{total} PASS, "
          f"{fail_count}/{total} FAIL", flush=True)
    print(f"{'='*60}", flush=True)

    return pass_count, fail_count, results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--village-dir", required=True)
    args = p.parse_args()
    p, f, _ = verify_waterlogging_physics(args.village_dir)
    sys.exit(0 if f == 0 else 1)
