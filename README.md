# HydroPath - Rural Drainage Automation Engine v2.0

**Physics-Aligned, Deterministic Village Drainage Analysis from Drone LiDAR**

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Project Overview](#2-project-overview)
3. [Problem Statement](#3-problem-statement)
4. [Target Villages](#4-target-villages)
5. [Technology Stack](#5-technology-stack)
6. [Directory Structure](#6-directory-structure)
7. [Pipeline Architecture](#7-pipeline-architecture)
8. [Engine Modules](#8-engine-modules)
9. [Configuration](#9-configuration)
10. [How to Run](#10-how-to-run)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Validation & Quality Assurance](#12-validation--quality-assurance)
13. [Known Limitations](#13-known-limitations)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Quick Start

### Prerequisites

Before running HydroPath, ensure you have:

- **Python 3.12+** installed
- **GDAL** with Python bindings
- **PDAL** command-line binary on PATH
- **WhiteboxTools** binary placed at `whitebox/WhiteboxTools_win_amd64/WBT/whitebox_tools.exe`

### Installation

```bash
# Clone or download the project
cd Rural_Drainage_Project

# Install Python dependencies
pip install -r requirements.txt

# Run product preflight checks (dependencies, binaries, model compatibility)
python product_healthcheck.py

# Verify WhiteboxTools is accessible
# On Windows, the binary should be at:
# whitebox/WhiteboxTools_win_amd64/WBT/whitebox_tools.exe
```

### Run a Single Village

```bash
# Process a single LAS/LAZ file
python run_pipeline_v2.py --input 00_raw_data/THANDALAM.las

# Process multiple files from a directory
python run_pipeline_v2.py --input 00_raw_data/
```

### AI Model Compatibility Note

- Runtime supports direct inference for `.onnx`, `.pkl`, `.joblib` models.
- PyTorch checkpoints (`.pth`, `.ckpt`) are supported through Myria3D model-native
    inference when the `myria3d` package is installed.
- If AI model loading is unavailable, pipeline safely falls back to SMRF ground
    classification.

### Expected Output

After successful execution, you'll find:
- `villages/<VILLAGE_NAME>/report/Village_Drainage_Report.pdf` - 3-page Panchayat-ready report
- `villages/<VILLAGE_NAME>/report/pipeline_manifest.json` - Execution metadata
- `villages/<VILLAGE_NAME>/drainage/` - Drainage network shapefiles
- `villages/<VILLAGE_NAME>/waterlogging/` - Risk classification rasters

---

## 2. Project Overview

HydroPath is an **end-to-end automated pipeline** that transforms raw **drone LiDAR point cloud data** (LAS/LAZ files) into actionable drainage design reports for rural Indian villages.

### What It Produces

From a single `.las` file, HydroPath generates:

1. **Digital Terrain Models (DTM)** at 1-meter resolution
2. **Hydrological Analysis** including:
   - Flow accumulation maps
   - Flow direction (D8 algorithm)
   - Slope analysis
   - Topographic Wetness Index (TWI)
3. **Waterlogging Susceptibility Maps** with risk classification:
   - LOW risk (green)
   - MEDIUM risk (yellow)
   - HIGH risk (red)
4. **Optimized Drainage Network Designs** using A* least-cost pathfinding
5. **Engineering Drain Design Parameters**:
   - Pipe sizing calculations
   - Trench dimensions
6. **Panchayat-Ready PDF Reports** with maps, priority tables, and recommendations

### The Complete Pipeline

```
Raw LAS/LAZ
    ↓
Ground Classification (SMRF)
    ↓
Digital Terrain Model (IDW)
    ↓
Hydrological Conditioning
    ↓
Waterlogging Risk Analysis (GMM)
    ↓
Drainage Network Routing (A*)
    ↓
Engineering Design (Rational Method)
    ↓
Validation & PDF Report
```

---

## 3. Problem Statement

Rural villages in India often suffer from **waterlogging and poor drainage** due to:
- Flat terrain with poor natural drainage
- Inadequate infrastructure development
- Lack of precise topographic data for planning
- Expensive and time-consuming traditional survey methods

### How HydroPath Solves This

1. **Drone-acquired LiDAR** provides high-resolution terrain data (1m)
2. **SMRF ground classification** separates ground from vegetation/buildings
3. **Physics-based terrain analysis** identifies waterlogging-prone areas
4. **AI-assisted routing** (A* pathfinding) designs optimal drain paths
5. **Deterministic, reproducible** outputs with fixed seeds and no randomness

---

## 4. Target Villages

HydroPath has been tested on villages with diverse terrain challenges:

| Village | State/Region | Terrain Type | Key Challenge |
|---------|-------------|--------------|---------------|
| **THANDALAM** | Tamil Nadu | Coastal Lowland | Needed CRS reprojection; elevation anomalies |
| **GANDHINAGAR_DIGLIPUR** | Andaman & Nicobar | Coastal Island | Extremely flat (mean elev 5.96m); routing challenges |
| **67169_5NKR_CHAKHIRASINGH** | Uttar Pradesh | Moderate | Sparse stream network |
| **DHAL_HOSHIARPUR** | Punjab | Flat Agricultural | Zero stream cells at default threshold |

### Supported Input Formats

- **LAS** - ASCII or binary LiDAR point cloud format
- **LAZ** - Compressed LiDAR format
- Automatic CRS detection and conversion to UTM

---

## 5. Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Point Cloud Processing** | PDAL | CRS detection, reprojection, SMRF classification, tiling, IDW interpolation |
| **Geospatial Rasters** | GDAL/OGR | Raster I/O, void filling, hillshade, coordinate transforms, shapefile writing |
| **Hydrological Modeling** | WhiteboxTools | Depression filling, D8 flow direction, flow accumulation, slope, convergence |
| **Scientific Computing** | NumPy + SciPy | GMM clustering, median filtering, distance transforms, A* search |
| **PDF Generation** | ReportLab | Panchayat-ready PDF reports with tables, maps, recommendations |
| **Visualization** | Matplotlib (Agg) | Map rendering, elevation profiles, drainage network plots |
| **Runtime** | Python 3.12+ | All orchestration and engine code |

### Explicit Dependency Avoidance

- **No scikit-learn** - GMM implemented from scratch in NumPy
- **No scikit-image** - Zhang-Suen thinning implemented in pure NumPy
- **No network dependencies** - Entire pipeline runs offline

---

## 6. Directory Structure

```
Rural_Drainage_Project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.yaml                  # Central configuration reference
├── run_pipeline_v2.py           # Main pipeline orchestrator
├── preprocess_village.py        # LAS preprocessing (CRS, Z-filter, tiling)
│
├── engine/                      # Core processing engine modules
│   ├── __init__.py
│   ├── config.py               # Configuration constants
│   ├── io_utils.py             # File I/O utilities with atomic writes
│   ├── terrain_generation.py   # SMRF ground extraction + IDW DTM
│   ├── hydrology_metrics.py    # TWI, flow accumulation, convergence
│   ├── waterlogging_model.py   # GMM-based risk classification
│   ├── watershed.py            # D8 flow direction, watershed delineation
│   ├── drainage_network.py     # A* routing + stream vectorization
│   ├── drain_design.py         # Rational Method pipe sizing
│   ├── verification.py         # Physical plausibility checks
│   ├── visualization_report.py # Multi-figure visualization reports
│   ├── export_basins.py        # Basin polygon export
│   ├── sensitivity_analysis.py # Parameter sensitivity sweeps
│   └── cross_village_validation.py # Cross-village validation
│
├── reporting/                   # Publication report generation
│   ├── __init__.py
│   ├── generate_final_report.py # 6-page competition PDF
│   └── generate_architecture_diagrams.py # System architecture PNGs
│
├── 00_raw_data/                 # Raw LAS/LAZ input files
│   ├── THANDALAM.las
│   ├── GANDHINAGAR_*.laz
│   └── ...
│
├── demo_small_las/              # Small demo files for testing
│   └── 67169_5NKR_CHAKHIRASINGH.las
│
├── villages/                    # Per-village processing outputs
│   ├── THANDALAM/
│   │   ├── input/              # Tiled LAS files
│   │   ├── terrain/            # DTM, slope, hillshade
│   │   ├── hydrology/          # TWI, flow accumulation
│   │   ├── waterlogging/       # Risk classification rasters
│   │   ├── drainage/           # Drainage network shapefiles
│   │   ├── report/             # PDF reports and manifest
│   │   └── logs/               # Processing logs
│   └── ...
│
├── reports/                     # Aggregated reports and artifacts
│   ├── AI_Drainage_Optimization_Report.pdf
│   ├── cross_village_statistics.csv
│   └── ...
│
├── logs/                        # Global processing logs
│
├── installer/                   # Installation utilities
│   ├── setup.py                # GUI installer wizard
│   ├── hydropath_installer.py  # Command-line installer
│   ├── build_installer_exe.py  # Build executable installer
│   └── README_INSTALL.txt      # Installation instructions
│
├── portable/                    # Portable executable build files
│   ├── hydropath_engine.py     # Frozen engine entry point
│   ├── build_portable_exe.py   # PyInstaller spec builder
│   └── README_PORTABLE.md      # Portable build instructions
│
├── whitebox/                    # WhiteboxTools binary
│   └── WhiteboxTools_win_amd64/
│       └── WBT/
│           └── whitebox_tools.exe
│
└── .gitignore                   # Git ignore rules
```

---

## 7. Pipeline Architecture

The pipeline runs **8 stages** sequentially per village. Fatal stages stop execution on failure; non-fatal stages continue.

### Stage-by-Stage Breakdown

| Stage | Name | Description | Fatal? |
|-------|------|-------------|--------|
| 1 | **Preprocess** | CRS detection → reproject → Z anomaly filter → tiling | Yes |
| 2 | **Terrain** | Per-tile SMRF → merge ground → IDW DTM → void fill → hydro conditioning | Yes |
| 3 | **Anomaly Fix** | Adaptive median filter, K*MAD outlier detection (window=51) | No |
| 4 | **Hydrology** | Fill depressions → D8 flow → flow accumulation → TWI → convergence | Yes |
| 5 | **Waterlogging** | TWI + dist-stream + rel-elev + convergence → GMM 3-class → HPF filter | Yes |
| 6 | **Drainage** | A* routing: gravity → pump → secondary outlets → retention basins | No |
| 7 | **Verification** | Physical plausibility self-checks | No |
| 8 | **Report** | PDF generation with maps, tables, recommendations | No |

### Detailed Stage Descriptions

#### Stage 1: Preprocess (`preprocess_village.py`)
1. **CRS Detection** - Uses `pdal info --summary` to extract EPSG code and bounds
2. **CRS Normalization** - Reprojects geographic CRS (EPSG:4326) to UTM automatically
3. **Z Anomaly Filtering** - Decimates to ~1M points, computes median + MAD, removes outliers outside `median ± 6*MAD`
4. **Tiling** - Splits into manageable chunks:
   - `filters.splitter` for < 100M points (in-memory)
   - `filters.crop` for > 100M points (streaming, constant memory)

#### Stage 2: Terrain Generation (`engine/terrain_generation.py`)
1. **Per-tile ground extraction**: SMRF classification (slope=0.2, window=10, threshold=0.45)
2. **Merge** all ground tiles into single point cloud
3. **IDW interpolation** at 1m resolution, 4m search radius
4. **Void filling** using GDAL `FillNodata` with 15m search distance
5. **Hydro conditioning** via WhiteboxTools: FillDepressions → Slope → D8Pointer

#### Stage 3: Anomaly Fix (inline in `run_pipeline_v2.py`)
- Applies `scipy.ndimage.median_filter` (51x51 window)
- Computes global MAD with 1.0m floor
- Flags cells where `value < local_median - 6.0 * MAD`
- Replaces flagged cells with local median values

#### Stage 4: Hydrology (`engine/hydrology_metrics.py`)
- Depression filling
- D8 flow direction computation
- Flow accumulation calculation
- Topographic Wetness Index (TWI)
- Relative elevation surface
- Convergence index

#### Stage 5: Waterlogging (`engine/waterlogging_model.py`)
1. Computes 7 terrain features (TWI, distance to stream, relative elevation, convergence, slope, flow accumulation, elevation)
2. **GMM clustering** (K=3) with full-covariance EM algorithm
3. **Physics-based cluster alignment** assigns LOW/MEDIUM/HIGH risk
4. **Hydrological Plausibility Filter (HPF)** demotes false positives:
   - Steep slopes (> 8°)
   - Elevated terrain (> 1.5m relative elevation)
   - Divergent flow (convergence > 10.0)

#### Stage 6: Drainage (`engine/drainage_network.py`)
**Four-pass A* routing with progressive fallback:**
1. **Pass 1**: Gravity to natural drains (max 1.0m cumulative uphill)
2. **Pass 2**: Pump to natural drains (up to 6.0m lift)
3. **Pass 3**: Gravity to secondary outlets (FA-derived, top 5%)
4. **Pass 4**: Pump to secondary outlets (final fallback)
5. Unroutable clusters → **retention basins**

**Cost function:** `0.6 * normalized_slope + 0.4 * normalized_convergence`

#### Stage 7: Verification (`engine/verification.py`)
- Validates HIGH-risk zones have lower slopes, higher TWI than LOW zones
- Confirms drainage routes obey gravity constraints
- Checks pump lifts stay within 6.0m cap

#### Stage 8: Report (`engine/report_generator.py`)
Generates a 3-page Panchayat Drainage Report:
1. **Page 1**: Executive summary with recommendations
2. **Page 2**: Color waterlogging risk map with basin overlay
3. **Page 3**: Priority zone table and drain design parameters

---

## 8. Engine Modules

### Core Processing Modules

| Module | Lines | Description |
|--------|-------|-------------|
| `terrain_generation.py` | ~11 KB | SMRF ground extraction + IDW DTM + hydro conditioning |
| `hydrology_metrics.py` | ~8.6 KB | TWI, flow accumulation, convergence, relative elevation |
| `waterlogging_model.py` | ~18 KB | GMM clustering + HPF filter (pure NumPy implementation) |
| `watershed.py` | ~10.8 KB | D8 flow direction, pour points, watershed delineation |
| `drainage_network.py` | ~41.6 KB | A* routing, stream vectorization, shapefile export |
| `drain_design.py` | ~18.2 KB | Rational Method pipe sizing, trench dimensions |
| `verification.py` | ~7.3 KB | Physical plausibility self-checks |
| `io_utils.py` | ~5 KB | Atomic file writes, NoData handling, format conversion |

### Analysis & Reporting Modules

| Module | Description |
|--------|-------------|
| `sensitivity_analysis.py` | Threshold sweeps, feature ablation studies |
| `cross_village_validation.py` | Cross-village stability checks, z-score anomaly detection |
| `visualization_report.py` | Multi-figure visualization reports |
| `export_basins.py` | Polygonize HIGH-risk clusters to shapefile |
| `config.py` | Central configuration constants |

---

## 9. Configuration

All parameters are defined in `config.yaml` and referenced in `engine/config.py`. Key parameters:

### Ground Classification (SMRF)
```yaml
smrf:
  slope: 0.2          # Slope threshold
  window: 10.0        # Window size in meters
  threshold: 0.45     # Classification threshold
  scalar: 1.25        # Scalar multiplier
```

### DTM & Interpolation
```yaml
dtm:
  resolution_m: 1.0   # 1-meter resolution
  idw_radius_m: 4.0   # 4m search radius
  void_fill_dist: 15  # 15m void fill search
  nodata: -9999.0
```

### Waterlogging Model
```yaml
waterlogging:
  w_twi: 0.35         # TWI weight
  w_convergence: 0.25 # Convergence weight
  w_rel_elev: -0.25   # Relative elevation weight
  hpf_slope_max: 6.0  # HPF slope threshold (degrees)
  hpf_rel_elev_max: -0.1  # HPF relative elevation threshold
```

### Drainage Routing
```yaml
drainage:
  max_pump_lift_m: 6.0    # Engineering limit
  max_routing_distance_m: 300  # Practical drain length
  min_cluster_area_m2: 50      # Minimum cluster size
  cost_weight_slope: 0.6       # Slope weight in cost function
  cost_weight_convergence: 0.4 # Convergence weight
```

---

## 10. How to Run

### Basic Usage

```bash
# Process a single village from raw LAS file
python run_pipeline_v2.py --input 00_raw_data/THANDALAM.las

# Process all villages in a directory
python run_pipeline_v2.py --input 00_raw_data/

# Process specific villages by name
python run_pipeline_v2.py --input 00_raw_data/ --villages THANDALAM GANDHINAGAR
```

### Advanced Options

```bash
# Resume from a specific stage (1-8)
python run_pipeline_v2.py --input 00_raw_data/THANDALAM.las --from-stage 5

# Reprocess existing village (skips preprocessing)
python run_pipeline_v2.py --reprocess --villages THANDALAM

# Migrate legacy village structure to new format
python run_pipeline_v2.py --reprocess --villages THANDALAM --migrate-legacy
```

### Programmatic Usage

```python
from run_pipeline_v2 import process_village

# Process a village programmatically
success, pdf_path, elapsed = process_village(
    village_name="THANDALAM",
    village_dir="villages/THANDALAM",
    from_stage=1
)

if success:
    print(f"Report generated: {pdf_path}")
```

### Output Files

For each processed village:

```
villages/<NAME>/
├── input/                    # Tiled LAS/LAZ files
├── terrain/
│   ├── village_dtm_clean.tif      # Cleaned DTM
│   ├── village_dtm_voidfill.tif   # Void-filled DTM
│   ├── filled_dtm.tif             # Hydro-filled DTM
│   ├── slope.tif                  # Slope raster
│   └── village_hillshade.tif      # Hillshade visualization
├── hydrology/
│   ├── flow_accumulation.tif      # Flow accumulation
│   ├── twi.tif                    # Topographic Wetness Index
│   ├── relative_elevation.tif     # Relative elevation
│   └── convergence_index.tif      # Convergence index
├── waterlogging/
│   ├── waterlogging_score.tif     # Continuous risk score
│   └── waterlogging_risk_class.tif # Classified risk (1=LOW, 2=MED, 3=HIGH)
├── drainage/
│   ├── optimized_drainage_network.shp  # Engineered drain routes
│   └── natural_drain_network.shp       # Natural stream network
├── report/
│   ├── Village_Drainage_Report.pdf     # 3-page Panchayat report
│   └── pipeline_manifest.json          # Execution metadata
└── logs/                     # Per-stage processing logs
```

---

## 11. Key Design Decisions

### Decision 1: Pure NumPy GMM (No scikit-learn)

**Why:** `scipy.stats.multivariate_normal.logpdf()` crashes with Windows SEH exceptions in BLAS DLL.

**Solution:** Full GMM EM algorithm implemented from scratch in NumPy (~350 lines):
- Cholesky decomposition for log-PDF
- Log-sum-exp trick for numerical stability
- 5 random initializations (seeds 42-46)
- Full covariance matrices with regularization

### Decision 2: Zhang-Suen Thinning (No scikit-image)

**Why:** Avoid adding scikit-image dependency for morphological thinning.

**Solution:** Classic Zhang-Suen two-sub-iteration thinning algorithm in pure NumPy (~100 lines).

### Decision 3: Elevation as CONSTRAINT, Not COST

**Why:** If elevation were a cost component, A* would avoid downhill paths.

**Solution:** Hard constraint of max 1.0m cumulative uphill for gravity, 6.0m for pump. Cost uses only slope (60%) and convergence (40%).

### Decision 4: Four-Pass Progressive Routing

**Why:** Single-pass routing failed on flat terrain (90% unreachable clusters).

**Evolution:**
| Version | Approach | Result |
|---------|----------|--------|
| v1 | Natural streams only | FAILED (unrealistic distances) |
| v2 | Conservative retention basins | Too conservative (78% retention) |
| v3 | Secondary outlets + 300m max | Practical (85% connectivity) |
| v4 (current) | Four-pass progressive | **93-100% connectivity** |

### Decision 5: Adaptive Stream Thresholding

**Why:** Fixed thresholds fail across villages with different terrain.

**Solution:** Density-based thresholding: search from P99.5 downward to find 0.05%-0.5% stream pixels. Fallback to P90 if needed.

### Decision 6: Atomic File Writes

**Why:** QGIS often holds file locks during development.

**Solution:** Write to `.tmp` file, then atomic rename. If rename fails, keep `.tmp` and check for it on read.

---

## 12. Validation & Quality Assurance

### Pre-Run Checks

1. **Environment validation** - Python version, dependencies, WhiteboxTools binary
2. **Input validation** - CRS detection, point count, bounds
3. **Memory safety** - Abort if >90% system RAM consumed

### Physics Validation

1. **Feature ordering** - HIGH zones must have lower slopes, higher TWI than LOW zones
2. **Drainage constraints** - Routes obey gravity (max uphill), pump lifts ≤ 6.0m
3. **Cluster sizing** - Minimum area threshold filters noise

### Cross-Village Validation

1. **Z-score anomaly detection** - Flags villages with unusual HIGH% (>2σ)
2. **Terrain classification** - Labels: COASTAL_LOWLAND, FLAT, MODERATE, HILLY
3. **Feasibility tiering** - GRAVITY_FEASIBLE (≥80%), MIXED (≥20%), ENGINEERED_REQUIRED

---

## 13. Known Limitations

1. **No building footprint data** - HPF filter uses heuristics; actual building polygons would improve accuracy
2. **Diffuse waterlogging** - Very flat villages (e.g., Dhal) need surface grading, not channel drainage
3. **Single-date LiDAR** - Seasonal variation not captured; requires repeat drone flights
4. **WhiteboxTools path** - Hardcoded to project root; needs config update for other systems
5. **Windows-only testing** - PDAL/GDAL paths use Windows conventions
6. **No rainfall intensity data** - Rational Method uses assumed design rainfall; IDF curves would improve estimates
7. **Large datasets** - Kadamtala (10.8 GB, 1.65B points) needs dedicated investigation

---

## 14. Troubleshooting

### Common Issues

#### "WhiteboxTools not found"
```bash
# Ensure binary exists at:
ls whitebox/WhiteboxTools_win_amd64/WBT/whitebox_tools.exe

# On Windows, add to PATH or update config.yaml
```

#### "PDAL command not found"
```bash
# Install PDAL from https://pdal.io/install.html
# Add PDAL bin directory to PATH
```

#### "GDAL import failed"
```bash
# Install GDAL with Python bindings:
pip install GDAL==$(gdal-config --version)

# Or use conda:
conda install -c conda-forge gdal
```

#### "Out of memory"
- For >100M point clouds, pipeline automatically uses streaming tiling
- Ensure 8GB+ RAM for large datasets
- Process smaller tiles individually if needed

#### "QGIS file lock"
- Close QGIS before running pipeline
- Pipeline uses atomic writes with `.tmp` fallback

### Getting Help

1. Check `logs/` directory for stage-specific error messages
2. Review `pipeline_manifest.json` for execution metadata
3. Run with `--from-stage` to isolate failing stage

---

*Engine version: 2.0.0 | Pipeline: Physics-Based Deterministic | Platform: Windows 11 | Python 3.12+*

*For questions or issues, please refer to the troubleshooting section or examine the processing logs.*