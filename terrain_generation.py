"""
engine/terrain_generation.py — Terrain Pipeline
=================================================
Per-tile SMRF ground extraction → merge → IDW DTM → void fill →
hydro-conditioning (FillDepressions, Slope, Hillshade).

Inputs:  <village_dir>/input/  (tiled .las/.laz files)
Outputs: <village_dir>/terrain/
  village_dtm_clean.tif  — production DTM (anomaly-fixed or void-filled)
  slope.tif              — slope in degrees
  filled_dtm.tif         — hydrologically conditioned DTM
  village_hillshade.tif  — visualization hillshade
"""
import os
import sys
import subprocess
import glob
import json
import time
import shutil
import yaml

import numpy as np
from osgeo import gdal as _gdal

_gdal.UseExceptions()

# ── Locked parameters ──────────────────────────────────────────
SMRF_SLOPE = 0.2
SMRF_WINDOW = 10.0
SMRF_THRESHOLD = 0.45
SMRF_SCALAR = 1.25
ELM_CELL = 10.0
ELM_THRESHOLD = 1.0
IDW_RESOLUTION = 1.0
IDW_RADIUS = 4.0
FILL_SEARCH_DIST = 15
NODATA = -9999.0

# ── Tool resolution ───────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_ai_settings():
    """Load AI point-classification settings from config.yaml."""
    cfg_path = os.path.join(PROJECT_DIR, "config.yaml")
    defaults = {
        "enable_ai": False,
        "model_path": "models/pointnet_drainage.onnx",
        "model_type": "pointnet",
        "class_names": ["Ground", "Vegetation", "Drainage", "Road"],
        "ground_labels": [0],
    }
    if not os.path.isfile(cfg_path):
        return defaults

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ai_cfg = cfg.get("ai_point_classification", {})
        if not isinstance(ai_cfg, dict):
            return defaults
        out = defaults.copy()
        out.update(ai_cfg)
        return out
    except Exception:
        return defaults


def _build_ground_extract_pipeline(input_las, output_las):
    """Build a PDAL pipeline that extracts final ground points only."""
    return {
        "pipeline": [
            {"type": "readers.las", "filename": input_las},
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {"type": "filters.elm", "cell": ELM_CELL, "threshold": ELM_THRESHOLD},
            {"type": "filters.range", "limits": "Classification![7:7]"},
            {"type": "writers.las", "filename": output_las},
        ]
    }

def _resolve(name):
    candidates = [
        shutil.which(name),
        os.path.join(os.path.dirname(sys.executable), "Library", "bin",
                     f"{name}.exe" if not name.endswith(".exe") else name),
        os.path.join(os.path.dirname(sys.executable), "Scripts",
                     f"{name}.exe" if not name.endswith(".exe") else name),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return name

PDAL_EXE = _resolve("pdal")
GDALDEM = _resolve("gdaldem")
WBT_EXE = os.path.join(PROJECT_DIR, "whitebox",
                        "WhiteboxTools_win_amd64", "WBT",
                        "whitebox_tools.exe")


def _file_ok(path, min_mb=0.001):
    return os.path.isfile(path) and os.path.getsize(path) / (1024*1024) >= min_mb


def _mb(path):
    return os.path.getsize(path) / (1024*1024)


def _run_cmd(cmd, desc):
    try:
        # Use CREATE_NO_WINDOW on Windows to prevent hidden dialog hangs
        kwargs = dict(capture_output=True, text=True, check=True)
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        if isinstance(cmd, str):
            kwargs["shell"] = True
        r = subprocess.run(cmd, **kwargs)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FATAL in {desc}: {e.stderr[:500]}", flush=True)
        raise RuntimeError(f"Command failed: {desc}: {e.stderr[:200]}")


def _banner(step, title):
    print(f"\n{'='*60}", flush=True)
    print(f"STEP {step} — {title}", flush=True)
    print(f"{'='*60}", flush=True)


def run(village_dir):
    """Run full terrain pipeline for a village.

    Reads tiles from village_dir/input/
    Writes terrain outputs to village_dir/terrain/
    """
    t_total = time.time()

    # Check input/ then fall back to 02_tiles/
    input_dir = os.path.join(village_dir, "input")
    alt = os.path.join(village_dir, "02_tiles")
    # Fall back to 02_tiles if input/ is empty or doesn't exist
    if not glob.glob(os.path.join(input_dir, "*_tile_*.*")):
        if os.path.isdir(alt) and glob.glob(os.path.join(alt, "*_tile_*.*")):
            input_dir = alt

    terrain_dir = os.path.join(village_dir, "terrain")
    ground_dir = os.path.join(terrain_dir, "_ground")
    os.makedirs(terrain_dir, exist_ok=True)
    os.makedirs(ground_dir, exist_ok=True)

    # Output paths
    MERGED_LAS = os.path.join(terrain_dir, "_merged_ground.las")
    RAW_DTM = os.path.join(terrain_dir, "village_dtm_raw.tif")
    VOIDFILL_DTM = os.path.join(terrain_dir, "village_dtm_voidfill.tif")
    CLEAN_DTM = os.path.join(terrain_dir, "village_dtm_clean.tif")
    HILLSHADE = os.path.join(terrain_dir, "village_hillshade.tif")
    FILLED_DTM = os.path.join(terrain_dir, "filled_dtm.tif")
    SLOPE_TIF = os.path.join(terrain_dir, "slope.tif")

    print("=" * 60, flush=True)
    print("  TERRAIN GENERATION", flush=True)
    print("=" * 60, flush=True)

    # ── STEP 1: Find tiles ──
    _banner(1, "Tile Selection")
    tiles = sorted(
        glob.glob(os.path.join(input_dir, "*_tile_*.las")) +
        glob.glob(os.path.join(input_dir, "*_tile_*.laz")))
    print(f"  Found {len(tiles)} tiles in {input_dir}", flush=True)
    if not tiles:
        print("  FATAL: No tiles found in input/ or 02_tiles/", flush=True)
        raise RuntimeError("No tiles found in input/ or 02_tiles/")

    # ── STEP 2: Per-tile ground extraction ──
    _banner(2, "Per-Tile Ground (SMRF + ELM)")
    t0 = time.time()
    ground_files = []
    skipped = 0

    ai_cfg = _load_ai_settings()
    enable_ai = bool(ai_cfg.get("enable_ai", False))
    model_path = ai_cfg.get("model_path")
    model_type = ai_cfg.get("model_type", "pointnet")
    class_names = ai_cfg.get("class_names", ["Ground", "Vegetation", "Drainage", "Road"])
    ground_labels = ai_cfg.get("ground_labels", [0])
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_DIR, model_path)

    classifier = None
    ai_ready = False
    if enable_ai:
        try:
            from engine.ai_point_classifier import AIPointClassifier
            classifier = AIPointClassifier(
                model_path=model_path,
                model_type=model_type,
                enable_ai=True,
                smrf_params={
                    "slope": SMRF_SLOPE,
                    "window": SMRF_WINDOW,
                    "threshold": SMRF_THRESHOLD,
                    "scalar": SMRF_SCALAR,
                },
                class_names=class_names,
                ground_labels=ground_labels,
            )
            status = classifier.get_status()
            ai_ready = bool(status.get("ai_enabled", False))
            if ai_ready:
                print(f"  AI classifier enabled: {model_type} ({model_path})", flush=True)
            else:
                print("  AI requested, but model not loaded. Using SMRF fallback.", flush=True)
        except Exception as e:
            print(f"  AI classifier unavailable ({e}). Using SMRF fallback.", flush=True)
            classifier = None

    n_ai = 0
    n_smrf = 0

    for i, las in enumerate(tiles):
        name = os.path.splitext(os.path.basename(las))[0]
        out = os.path.join(ground_dir, f"ground_{name}.las")

        if _file_ok(out):
            ground_files.append(out)
            skipped += 1
            continue

        print(f"  [{i+1:2d}/{len(tiles)}] {name}: ", end="", flush=True)
        if enable_ai and classifier is not None:
            tmp_classified = os.path.join(terrain_dir, f"_tmp_cls_{name}.las")
            try:
                _, used_ai = classifier.classify_las_file(las, tmp_classified, use_ai=True)
                mode = "AI" if used_ai else "SMRF-FB"
                if used_ai:
                    n_ai += 1
                else:
                    n_smrf += 1

                pipeline = _build_ground_extract_pipeline(tmp_classified, out)
                pf = os.path.join(terrain_dir, "_tmp_pipeline.json")
                with open(pf, "w", encoding="utf-8") as f:
                    json.dump(pipeline, f, indent=2)
                _run_cmd(f'"{PDAL_EXE}" pipeline "{pf}"', f"Ground extract {name}")
                if os.path.exists(pf):
                    os.remove(pf)
                if os.path.exists(tmp_classified):
                    os.remove(tmp_classified)
                print(f"{mode} -> GROUND", end="", flush=True)
            except Exception:
                if os.path.exists(tmp_classified):
                    try:
                        os.remove(tmp_classified)
                    except OSError:
                        pass
                pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": las},
                        {"type": "filters.outlier", "method": "statistical",
                         "mean_k": 8, "multiplier": 2.0},
                        {"type": "filters.smrf", "slope": SMRF_SLOPE,
                         "window": SMRF_WINDOW, "threshold": SMRF_THRESHOLD,
                         "scalar": SMRF_SCALAR},
                        {"type": "filters.range", "limits": "Classification[2:2]"},
                        {"type": "filters.elm", "cell": ELM_CELL,
                         "threshold": ELM_THRESHOLD},
                        {"type": "filters.range", "limits": "Classification![7:7]"},
                        {"type": "writers.las", "filename": out}
                    ]
                }
                pf = os.path.join(terrain_dir, "_tmp_pipeline.json")
                with open(pf, 'w') as f:
                    json.dump(pipeline, f, indent=2)
                _run_cmd(f'"{PDAL_EXE}" pipeline "{pf}"', f"SMRF {name}")
                if os.path.exists(pf):
                    os.remove(pf)
                n_smrf += 1
                print("SMRF-FB2", end="", flush=True)
        else:
            pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": las},
                    {"type": "filters.outlier", "method": "statistical",
                     "mean_k": 8, "multiplier": 2.0},
                    {"type": "filters.smrf", "slope": SMRF_SLOPE,
                     "window": SMRF_WINDOW, "threshold": SMRF_THRESHOLD,
                     "scalar": SMRF_SCALAR},
                    {"type": "filters.range", "limits": "Classification[2:2]"},
                    {"type": "filters.elm", "cell": ELM_CELL,
                     "threshold": ELM_THRESHOLD},
                    {"type": "filters.range", "limits": "Classification![7:7]"},
                    {"type": "writers.las", "filename": out}
                ]
            }
            pf = os.path.join(terrain_dir, "_tmp_pipeline.json")
            with open(pf, 'w') as f:
                json.dump(pipeline, f, indent=2)
            _run_cmd(f'"{PDAL_EXE}" pipeline "{pf}"', f"SMRF {name}")
            if os.path.exists(pf):
                os.remove(pf)
            n_smrf += 1

        if _file_ok(out):
            ground_files.append(out)
            print(f" DONE ({_mb(out):.1f} MB)", flush=True)
        else:
            print("FAIL", flush=True)

    print(f"  Total: {len(ground_files)} OK ({skipped} cached), "
          f"{time.time()-t0:.0f}s", flush=True)
    if enable_ai:
        print(f"  AI usage summary: AI={n_ai}, SMRF/Fallback={n_smrf}, ai_ready={ai_ready}", flush=True)

    # ── STEP 3: Merge + IDW DTM ──
    _banner(3, "Merge Ground + IDW DTM")

    if _file_ok(MERGED_LAS, min_mb=1.0):
        print(f"  Merged LAS: cached ({_mb(MERGED_LAS):.0f} MB)", flush=True)
    else:
        t0 = time.time()
        merge_pipe = {
            "pipeline": [
                *[{"type": "readers.las", "filename": g}
                  for g in ground_files],
                {"type": "filters.merge"},
                {"type": "writers.las", "filename": MERGED_LAS}
            ]
        }
        pf = os.path.join(terrain_dir, "_tmp_merge.json")
        with open(pf, 'w') as f:
            json.dump(merge_pipe, f, indent=2)
        _run_cmd(f'"{PDAL_EXE}" pipeline "{pf}"', "Merge")
        os.remove(pf) if os.path.exists(pf) else None
        print(f"  Merged: {_mb(MERGED_LAS):.0f} MB ({time.time()-t0:.0f}s)",
              flush=True)

    if _file_ok(RAW_DTM, min_mb=0.1):
        print(f"  Raw DTM: cached ({_mb(RAW_DTM):.1f} MB)", flush=True)
    else:
        t0 = time.time()
        dtm_pipe = {
            "pipeline": [
                {"type": "readers.las", "filename": MERGED_LAS},
                {"type": "writers.gdal", "filename": RAW_DTM,
                 "resolution": IDW_RESOLUTION, "output_type": "idw",
                 "radius": IDW_RADIUS, "data_type": "float32",
                 "nodata": -9999,
                 "gdalopts": ["COMPRESS=LZW", "TILED=YES"]}
            ]
        }
        pf = os.path.join(terrain_dir, "_tmp_dtm.json")
        with open(pf, 'w') as f:
            json.dump(dtm_pipe, f, indent=2)
        _run_cmd(f'"{PDAL_EXE}" pipeline "{pf}"', "IDW")
        os.remove(pf) if os.path.exists(pf) else None
        print(f"  Raw DTM: {_mb(RAW_DTM):.1f} MB ({time.time()-t0:.0f}s)",
              flush=True)

    # ── STEP 4: Void fill ──
    _banner(4, "Controlled Void Fill")

    if _file_ok(VOIDFILL_DTM, min_mb=0.1):
        print(f"  Void-filled: cached ({_mb(VOIDFILL_DTM):.1f} MB)",
              flush=True)
    else:
        t0 = time.time()
        src = _gdal.Open(RAW_DTM, _gdal.GA_ReadOnly)
        drv = _gdal.GetDriverByName("GTiff")
        dst = drv.CreateCopy(VOIDFILL_DTM, src,
                             options=["COMPRESS=LZW", "TILED=YES"])
        src = None
        band = dst.GetRasterBand(1)
        _gdal.FillNodata(targetBand=band, maskBand=None,
                         maxSearchDist=FILL_SEARCH_DIST,
                         smoothingIterations=0)
        band.FlushCache()
        band.ComputeStatistics(False)
        dst = None
        print(f"  Void-filled: {_mb(VOIDFILL_DTM):.1f} MB "
              f"({time.time()-t0:.1f}s)", flush=True)

    # Resolve active DTM
    if _file_ok(CLEAN_DTM, min_mb=0.1):
        active = CLEAN_DTM
        print(f"  Active DTM: village_dtm_clean.tif", flush=True)
    elif _file_ok(VOIDFILL_DTM):
        active = VOIDFILL_DTM
        # Copy to clean name for downstream consumers
        shutil.copy2(VOIDFILL_DTM, CLEAN_DTM)
        active = CLEAN_DTM
        print(f"  Active DTM: copied voidfill → clean", flush=True)
    else:
        shutil.copy2(RAW_DTM, CLEAN_DTM)
        active = CLEAN_DTM
        print(f"  Active DTM: copied raw → clean (fallback)", flush=True)

    # ── STEP 5: FillDepressions + Slope ──
    _banner(5, "Hydro Conditioning (FillDepressions, Slope)")

    if _file_ok(FILLED_DTM, min_mb=0.1):
        print(f"  FillDepressions: cached", flush=True)
    else:
        t0 = time.time()
        _run_cmd(
            f'"{WBT_EXE}" -r=FillDepressions '
            f'-i="{active}" -o="{FILLED_DTM}"',
            "FillDepressions")
        print(f"  FillDepressions: {time.time()-t0:.1f}s", flush=True)

    if _file_ok(SLOPE_TIF, min_mb=0.01):
        print(f"  Slope: cached", flush=True)
    else:
        t0 = time.time()
        _run_cmd(
            f'"{WBT_EXE}" -r=Slope -i="{FILLED_DTM}" '
            f'-o="{SLOPE_TIF}" --units=degrees',
            "Slope")
        print(f"  Slope: {time.time()-t0:.1f}s", flush=True)

    # ── STEP 6: Hillshade ──
    _banner(6, "Hillshade")
    if _file_ok(HILLSHADE, min_mb=0.01):
        print(f"  Hillshade: cached", flush=True)
    else:
        _run_cmd(
            f'"{GDALDEM}" hillshade "{active}" "{HILLSHADE}" -of GTiff',
            "Hillshade")
        print(f"  Hillshade: done", flush=True)

    # ── Cleanup: remove temp ground files ──
    # Per contract: "No intermediate tiles saved permanently"
    # Keep ground dir for resume; nuke only _tmp files
    for f in glob.glob(os.path.join(terrain_dir, "_tmp_*")):
        try:
            os.remove(f)
        except OSError:
            pass

    elapsed = time.time() - t_total
    print(f"\n  Terrain total: {elapsed:.0f}s ({elapsed/60:.1f} min)",
          flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--village-dir", required=True)
    args = p.parse_args()
    run(args.village_dir)
