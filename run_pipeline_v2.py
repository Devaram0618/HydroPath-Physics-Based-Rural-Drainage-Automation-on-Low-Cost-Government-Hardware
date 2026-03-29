#!/usr/bin/env python3
"""
run_pipeline_v2.py — Physics-Based Village Drainage Pipeline (v2)
==================================================================
Deterministic LAS-to-PDF pipeline using engine/ modules.

Folder structure per village:
    villages/<VILLAGE>/
        input/          tiled .las/.laz point clouds
        terrain/        DTM, slope, hillshade, filled_dtm
        hydrology/      TWI, flow_accumulation, relative_elevation, …
        waterlogging/   waterlogging_score.tif, waterlogging_risk_class.tif
        drainage/       optimized_drainage_network.shp, natural_drain_network.shp
        report/         Village_Drainage_Report.pdf, summary_metrics.json
        logs/           per-stage log files

Stages:
    1  Preprocess      CRS detection, Z filtering, tiling
    2  Terrain         SMRF ground + IDW DTM + hydro conditioning
    3  Anomaly Fix     Adaptive Z outlier correction (optional)
    4  Hydrology       TWI, flow accumulation, relative elevation, convergence
    5  Waterlogging    Physics-based susceptibility scoring (deterministic)
    6  Drainage        A* routing + stream vectorization
    7  Verification    Self-check of physics constraints
    8  Report          3-page Panchayat PDF

Usage:
    python run_pipeline_v2.py --input ./village_las/
    python run_pipeline_v2.py --input ./THANDALAM.las
    python run_pipeline_v2.py --reprocess --villages THANDALAM
    python run_pipeline_v2.py --reprocess --from-stage 5
"""
import os
import sys
import subprocess
import time
import json
import re
import glob
import argparse
import shutil

import numpy as np
from scipy.ndimage import median_filter
from osgeo import gdal

gdal.UseExceptions()

# ── Paths ──────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VILLAGES_DIR = os.path.join(PROJECT_DIR, "villages")
PYTHON = sys.executable

# ── Engine imports ─────────────────────────────────────────────
sys.path.insert(0, PROJECT_DIR)
from engine import terrain_generation
from engine import hydrology_metrics
from engine import waterlogging_model
from engine import drainage_network
from engine import verification
from engine import report_generator

# ── Legacy script paths (preprocess only) ──────────────────────
PREPROCESS_SCRIPT = os.path.join(PROJECT_DIR, "preprocess_village.py")

# ── Blocklist ──────────────────────────────────────────────────
BLOCKLIST = ["64334_2H (REFLIGHT)_POINT CLOUD.LAS"]

NODATA = -9999.0

# ── Stage definitions ──────────────────────────────────────────
STAGE_NAMES = {
    1: "Preprocess",
    2: "Terrain",
    3: "Anomaly Fix",
    4: "Hydrology",
    5: "Waterlogging",
    6: "Drainage",
    7: "Verification",
    8: "Report",
}
TOTAL_STAGES = len(STAGE_NAMES)


# ================================================================
# Helpers
# ================================================================
def _normalize_name(filename):
    """Convert raw LAS filename to a standardised village name."""
    name = os.path.splitext(filename)[0]
    for suffix in ("_group1_densified_point_cloud",
                   "_densified_point_cloud",
                   "_point_cloud",
                   "_POINT CLOUD",
                   " POINT CLOUD"):
        idx = name.lower().find(suffix.lower())
        if idx > 0:
            name = name[:idx]
    name = re.sub(r'\s*\([^)]*\)\s*', '', name)
    name = name.replace(" ", "_").replace("&", "_").replace("-", "_")
    name = re.sub(r'_+', '_', name).strip("_").upper()
    return name


def _discover_inputs(input_path, village_filter=None):
    """Find LAS/LAZ files. Returns [(name, raw_path)]."""
    villages = []
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in (".las", ".laz"):
            name = _normalize_name(os.path.basename(input_path))
            villages.append((name, os.path.abspath(input_path)))
    elif os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            ext = os.path.splitext(f)[1].lower()
            if ext not in (".las", ".laz"):
                continue
            if f in BLOCKLIST:
                continue
            name = _normalize_name(f)
            if village_filter and name not in [v.upper() for v in village_filter]:
                continue
            villages.append((name, os.path.abspath(os.path.join(input_path, f))))
    return villages


def _discover_existing(village_filter=None):
    """Find already-processed villages (for --reprocess).
    Returns [(name, village_dir)]."""
    villages = []

    # Check project root (THANDALAM special case)
    tiles_root = os.path.join(PROJECT_DIR, "02_tiles")
    if os.path.isdir(tiles_root) and glob.glob(os.path.join(tiles_root, "*_tile_*.las")):
        name = "THANDALAM"
        if not village_filter or name in [v.upper() for v in village_filter]:
            villages.append((name, PROJECT_DIR))

    # Check villages/ subdirectories (new structure)
    if os.path.isdir(VILLAGES_DIR):
        for d in sorted(os.listdir(VILLAGES_DIR)):
            vdir = os.path.join(VILLAGES_DIR, d)
            if not os.path.isdir(vdir):
                continue
            # Accept if input/ or 02_tiles/ exists with tiles
            for sub in ("input", "02_tiles"):
                td = os.path.join(vdir, sub)
                if os.path.isdir(td) and (
                        glob.glob(os.path.join(td, "*_tile_*.las")) or
                        glob.glob(os.path.join(td, "*_tile_*.laz"))):
                    name = d.upper()
                    if not village_filter or name in [v.upper() for v in village_filter]:
                        villages.append((name, vdir))
                    break
    return villages


def _has_tiles(village_dir):
    for sub in ("input", "02_tiles"):
        td = os.path.join(village_dir, sub)
        if os.path.isdir(td) and glob.glob(os.path.join(td, "*_tile_*.*")):
            return True
    return False


def _progress(stage_num, label, status, elapsed=None):
    bar = "█" * stage_num + "░" * (TOTAL_STAGES - stage_num)
    t = ""
    if elapsed is not None:
        t = f" ({elapsed:.0f}s)" if elapsed < 60 else f" ({elapsed/60:.1f}min)"
    print(f"  [{bar}] {stage_num}/{TOTAL_STAGES}  {label:<30s} {status}{t}",
          flush=True)


def _find_pdf(village_dir):
    rdir = os.path.join(village_dir, "report")
    if os.path.isdir(rdir):
        for f in os.listdir(rdir):
            if f.lower().endswith(".pdf"):
                return os.path.join(rdir, f)
    return None


# ================================================================
# Anomaly fix (inline — simple median-filter outlier correction)
# ================================================================
def _anomaly_fix(village_dir):
    """Fix extreme Z outliers in the DTM using adaptive local-median."""
    terrain_dir = os.path.join(village_dir, "terrain")
    # Prefer voidfill, fall back to raw DTM
    voidfill = os.path.join(terrain_dir, "village_dtm_voidfill.tif")
    clean = os.path.join(terrain_dir, "village_dtm_clean.tif")

    if not os.path.isfile(voidfill):
        print("  No voidfill DTM found — skipping anomaly fix", flush=True)
        return True

    if os.path.isfile(clean):
        print("  village_dtm_clean.tif exists — skipping anomaly fix", flush=True)
        return True

    WINDOW = 51
    K_MAD = 6.0

    ds = gdal.Open(voidfill, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    valid = arr != nd if nd is not None else np.ones_like(arr, dtype=bool)
    if valid.sum() == 0:
        print("  WARNING: DTM is entirely NODATA", flush=True)
        return False

    gmed = np.median(arr[valid])
    gmad = np.median(np.abs(arr[valid] - gmed))
    gmad = max(gmad, 1.0)

    work = arr.copy()
    work[~valid] = gmed
    local_med = median_filter(work, size=WINDOW)
    outlier = valid & (arr < local_med - K_MAD * gmad)
    n_fix = int(outlier.sum())
    print(f"  Anomaly fix: {n_fix} outlier pixels corrected", flush=True)

    result = arr.copy()
    result[outlier] = local_med[outlier]

    # Write clean DTM
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(clean, result.shape[1], result.shape[0], 1,
                        gdal.GDT_Float32,
                        ["COMPRESS=LZW", "TILED=YES"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    ob = out_ds.GetRasterBand(1)
    ob.SetNoDataValue(NODATA)
    ob.WriteArray(result.astype(np.float32))
    ob.FlushCache()
    out_ds = None

    print(f"  Wrote {clean}", flush=True)
    return True


# ================================================================
# Preprocess (subprocess — uses legacy preprocess_village.py)
# ================================================================
def _run_preprocess(raw_path, village_dir):
    """Run preprocessing — direct call if possible, subprocess fallback."""
    # Try direct import first (works in frozen exe and normal Python)
    try:
        import preprocess_village as pv
        ok = pv.preprocess(raw_path, village_dir)
    except Exception:
        # Fallback: subprocess (only works with unfrozen Python)
        if not os.path.isfile(PREPROCESS_SCRIPT):
            print(f"  WARNING: preprocess_village.py not found", flush=True)
            return False

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["DRAINAGE_BASE_DIR"] = village_dir

        cmd = [PYTHON, PREPROCESS_SCRIPT,
               "--input", raw_path,
               "--output", village_dir]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=3600)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  Preprocess FAILED ({elapsed:.0f}s)", flush=True)
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"    {line[:120]}", flush=True)
            return False
        ok = True

    # Copy tiles to input/ if they went to 02_tiles/
    tiles_dir = os.path.join(village_dir, "02_tiles")
    input_dir = os.path.join(village_dir, "input")
    if os.path.isdir(tiles_dir):
        tile_files = glob.glob(os.path.join(tiles_dir, "*_tile_*.*"))
        if tile_files and not glob.glob(os.path.join(input_dir, "*_tile_*.*")):
            os.makedirs(input_dir, exist_ok=True)
            for f in tile_files:
                dst = os.path.join(input_dir, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy2(f, dst)

    return ok


# ================================================================
# Migrate legacy THANDALAM structure
# ================================================================
def migrate_legacy_village(legacy_dir, village_dir):
    """Copy legacy numbered-directory outputs into new named-directory structure.

    Maps:
        02_tiles/* → input/*
        05_dtm/full_village/village_dtm_clean.tif → terrain/
        05_dtm/full_village/slope.tif → terrain/
        05_dtm/full_village/filled_dtm.tif → terrain/
        05_dtm/full_village/village_hillshade.tif → terrain/
        05_dtm/full_village/flow_accumulation.tif → hydrology/
        07_waterlogging/twi.tif → hydrology/
        07_waterlogging/relative_elevation.tif → hydrology/
        07_waterlogging/convergence_index.tif → hydrology/
        07_waterlogging/dist_to_stream.tif → hydrology/
    """
    print("\n  Migrating legacy village structure...", flush=True)

    for sub in ("input", "terrain", "hydrology", "waterlogging",
                "drainage", "report", "logs"):
        os.makedirs(os.path.join(village_dir, sub), exist_ok=True)

    copies = [
        # (src relative to legacy_dir, dst relative to village_dir)
        # Tiles
        ("02_tiles", "input"),
        # Terrain
        ("05_dtm/full_village/village_dtm_clean.tif", "terrain/village_dtm_clean.tif"),
        ("05_dtm/full_village/village_dtm_voidfill.tif", "terrain/village_dtm_voidfill.tif"),
        ("05_dtm/full_village/slope.tif", "terrain/slope.tif"),
        ("05_dtm/full_village/filled_dtm.tif", "terrain/filled_dtm.tif"),
        ("05_dtm/full_village/village_hillshade.tif", "terrain/village_hillshade.tif"),
        # Hydrology
        ("05_dtm/full_village/flow_accumulation.tif", "hydrology/flow_accumulation.tif"),
        ("07_waterlogging/twi.tif", "hydrology/twi.tif"),
        ("07_waterlogging/relative_elevation.tif", "hydrology/relative_elevation.tif"),
        ("07_waterlogging/convergence_index.tif", "hydrology/convergence_index.tif"),
        ("07_waterlogging/dist_to_stream.tif", "hydrology/dist_to_stream.tif"),
    ]

    n_copied = 0
    for src_rel, dst_rel in copies:
        src = os.path.join(legacy_dir, src_rel)
        dst = os.path.join(village_dir, dst_rel)

        # Directory → copy all files in it
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                sf = os.path.join(src, f)
                df = os.path.join(dst, f)
                if os.path.isfile(sf) and not os.path.exists(df):
                    shutil.copy2(sf, df)
                    n_copied += 1
        # Single file
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                n_copied += 1
        else:
            print(f"    SKIP (not found): {src_rel}", flush=True)

    print(f"  Migrated {n_copied} files → {village_dir}", flush=True)
    return True


# ================================================================
# Process a single village
# ================================================================
def process_village(village_name, raw_path=None, village_dir=None,
                    from_stage=1, legacy_dir=None):
    """Run physics-based pipeline for one village.

    Returns: (success, pdf_path, total_seconds)
    """
    if village_dir is None:
        village_dir = os.path.join(VILLAGES_DIR, village_name)

    # Create output directories
    for sub in ("input", "terrain", "hydrology", "waterlogging",
                "drainage", "report", "logs"):
        os.makedirs(os.path.join(village_dir, sub), exist_ok=True)

    # Migrate legacy structure if requested
    if legacy_dir and legacy_dir != village_dir:
        migrate_legacy_village(legacy_dir, village_dir)

    t_total = time.time()
    stage_times = {}
    completed = 0

    print(f"\n  Village directory: {village_dir}")
    print(f"  Stages: {TOTAL_STAGES}\n")

    # ── STAGE 1: Preprocess ──
    s = 1
    if s >= from_stage:
        if raw_path and not _has_tiles(village_dir):
            _progress(s, STAGE_NAMES[s], "RUNNING...")
            t0 = time.time()
            ok = _run_preprocess(raw_path, village_dir)
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            if ok:
                _progress(s, STAGE_NAMES[s], "OK", elapsed)
                completed += 1
            else:
                _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
                return False, None, time.time() - t_total
        else:
            _progress(s, STAGE_NAMES[s], "SKIP")
            stage_times[s] = 0
            completed += 1
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 2: Terrain ──
    s = 2
    if s >= from_stage:
        clean_dtm = os.path.join(village_dir, "terrain", "village_dtm_clean.tif")
        filled_dtm = os.path.join(village_dir, "terrain", "filled_dtm.tif")
        if os.path.isfile(clean_dtm) and os.path.isfile(filled_dtm):
            _progress(s, STAGE_NAMES[s], "SKIP (outputs exist)")
            stage_times[s] = 0
            completed += 1
        else:
            _progress(s, STAGE_NAMES[s], "RUNNING...")
            t0 = time.time()
            try:
                terrain_generation.run(village_dir)
                elapsed = time.time() - t0
                stage_times[s] = elapsed
                _progress(s, STAGE_NAMES[s], "OK", elapsed)
                completed += 1
            except Exception as e:
                elapsed = time.time() - t0
                stage_times[s] = elapsed
                _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
                print(f"         {str(e)[:120]}", flush=True)
                return False, None, time.time() - t_total
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 3: Anomaly Fix ──
    s = 3
    if s >= from_stage:
        _progress(s, STAGE_NAMES[s], "RUNNING...")
        t0 = time.time()
        _anomaly_fix(village_dir)
        elapsed = time.time() - t0
        stage_times[s] = elapsed
        _progress(s, STAGE_NAMES[s], "OK", elapsed)
        completed += 1
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 4: Hydrology ──
    s = 4
    if s >= from_stage:
        twi = os.path.join(village_dir, "hydrology", "twi.tif")
        rel_elev = os.path.join(village_dir, "hydrology", "relative_elevation.tif")
        if os.path.isfile(twi) and os.path.isfile(rel_elev):
            _progress(s, STAGE_NAMES[s], "SKIP (outputs exist)")
            stage_times[s] = 0
            completed += 1
        else:
            _progress(s, STAGE_NAMES[s], "RUNNING...")
            t0 = time.time()
            try:
                hydrology_metrics.run(village_dir)
                elapsed = time.time() - t0
                stage_times[s] = elapsed
                _progress(s, STAGE_NAMES[s], "OK", elapsed)
                completed += 1
            except Exception as e:
                elapsed = time.time() - t0
                stage_times[s] = elapsed
                _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
                print(f"         {str(e)[:120]}", flush=True)
                return False, None, time.time() - t_total
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 5: Waterlogging ──
    s = 5
    if s >= from_stage:
        _progress(s, STAGE_NAMES[s], "RUNNING...")
        t0 = time.time()
        try:
            waterlogging_model.run(village_dir)
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "OK", elapsed)
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
            print(f"         {str(e)[:120]}", flush=True)
            return False, None, time.time() - t_total
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 6: Drainage ──
    s = 6
    if s >= from_stage:
        _progress(s, STAGE_NAMES[s], "RUNNING...")
        t0 = time.time()
        try:
            drainage_network.run(village_dir)
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "OK", elapsed)
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
            print(f"         {str(e)[:120]}", flush=True)
            # Non-fatal: drainage routing can partially fail
            completed += 1
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 7: Verification ──
    s = 7
    if s >= from_stage:
        _progress(s, STAGE_NAMES[s], "RUNNING...")
        t0 = time.time()
        try:
            p, f, results = verification.verify_waterlogging_physics(village_dir)
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            status = "OK" if f == 0 else f"WARN ({f} fail)"
            _progress(s, STAGE_NAMES[s], status, elapsed)
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
            print(f"         {str(e)[:120]}", flush=True)
            completed += 1  # non-fatal
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    # ── STAGE 8: Report ──
    s = 8
    if s >= from_stage:
        _progress(s, STAGE_NAMES[s], "RUNNING...")
        t0 = time.time()
        try:
            report_generator.run(village_dir)
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "OK", elapsed)
            completed += 1
        except Exception as e:
            elapsed = time.time() - t0
            stage_times[s] = elapsed
            _progress(s, STAGE_NAMES[s], "FAILED", elapsed)
            print(f"         {str(e)[:120]}", flush=True)
            completed += 1  # non-fatal
    else:
        _progress(s, STAGE_NAMES[s], "SKIP")
        stage_times[s] = 0
        completed += 1

    total_time = time.time() - t_total
    pdf_path = _find_pdf(village_dir)

    # Write pipeline manifest
    manifest = {
        "village": village_name,
        "village_dir": village_dir,
        "pipeline_version": "2.0.0",
        "model": "physics_based_deterministic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_s": round(total_time, 1),
        "stages_completed": completed,
        "stages_total": TOTAL_STAGES,
        "stage_times": {STAGE_NAMES[k]: round(v, 1) for k, v in stage_times.items()},
        "pdf_report": pdf_path,
        "success": completed == TOTAL_STAGES,
    }
    manifest_path = os.path.join(village_dir, "report", "pipeline_manifest.json")
    with open(manifest_path, "w") as f_out:
        json.dump(manifest, f_out, indent=2)

    return completed == TOTAL_STAGES, pdf_path, total_time


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Physics-Based Village Drainage Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_pipeline_v2.py --input ./village_las/
  python run_pipeline_v2.py --reprocess --villages THANDALAM
  python run_pipeline_v2.py --reprocess --from-stage 5
""")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to LAS/LAZ file or folder")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-run on already-tiled villages")
    parser.add_argument("--villages", nargs="*", default=None,
                        help="Process only these villages")
    parser.add_argument("--from-stage", type=int, default=1,
                        help="Resume from this stage number (1-8)")
    parser.add_argument("--migrate-legacy", action="store_true",
                        help="Migrate legacy numbered dirs to new structure")
    args = parser.parse_args()

    if not args.input and not args.reprocess:
        parser.error("Provide --input <path> or --reprocess")

    # ── Banner ──
    print()
    print("=" * 60)
    print("  VILLAGE DRAINAGE PIPELINE v2.0")
    print("  Physics-Based Deterministic Model")
    print("  LiDAR Point Cloud → Drainage Report (PDF)")
    print("=" * 60)

    # ── Discover villages ──
    if args.reprocess:
        villages = _discover_existing(args.villages)
        mode = "reprocess"
    else:
        villages = _discover_inputs(args.input, args.villages)
        mode = "fresh"

    if not villages:
        print("\n  No villages found.")
        sys.exit(1)

    print(f"\n  Mode: {mode}")
    print(f"  Villages: {len(villages)}")
    for name, path in villages:
        print(f"    {name:<30s} {path}")

    t_global = time.time()
    results = []

    for village_name, path in villages:
        print(f"\n{'=' * 60}")
        print(f"  PROCESSING: {village_name}")
        print(f"{'=' * 60}")

        # Determine village directory
        if mode == "fresh":
            vdir = os.path.join(VILLAGES_DIR, village_name)
            legacy = None
            success, pdf, elapsed = process_village(
                village_name, raw_path=path, village_dir=vdir,
                from_stage=args.from_stage)
        else:
            # For reprocessing, check if this is legacy root-level village
            if path == PROJECT_DIR:
                # Legacy THANDALAM at project root
                vdir = os.path.join(VILLAGES_DIR, village_name)
                legacy = PROJECT_DIR if args.migrate_legacy else None
                success, pdf, elapsed = process_village(
                    village_name, village_dir=vdir,
                    from_stage=args.from_stage,
                    legacy_dir=legacy)
            else:
                success, pdf, elapsed = process_village(
                    village_name, village_dir=path,
                    from_stage=args.from_stage)

        results.append({
            "village": village_name,
            "success": success,
            "pdf": pdf,
            "time": elapsed,
        })

    # ── Final summary ──
    total_time = time.time() - t_global
    n_ok = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_ok

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE (v2.0)", flush=True)
    print(f"{'=' * 60}")
    print(f"\n  Villages processed: {len(results)}")
    print(f"  Successful: {n_ok}")
    if n_fail:
        print(f"  Failed: {n_fail}")
    t_str = f"{total_time:.0f}s" if total_time < 60 else f"{total_time/60:.1f}min"
    print(f"  Total time: {t_str}")

    print(f"\n  Reports:")
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        if r["pdf"] and os.path.isfile(r["pdf"]):
            sz = os.path.getsize(r["pdf"]) / (1024 * 1024)
            print(f"    [{status}] {r['village']}: {r['pdf']} ({sz:.1f} MB)")
        else:
            print(f"    [{status}] {r['village']}: No PDF generated")
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
