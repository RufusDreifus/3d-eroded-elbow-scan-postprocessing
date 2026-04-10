# 3D Eroded Elbow Scan Post-Processing

Python tools for post-processing and analysis of 3D scanned eroded pipe elbows from STL geometry.

---

## Overview

This repository contains Python scripts developed for geometric and statistical analysis of eroded pipe elbows using STL scan data.

The current workflow supports:

- wall-thickness evaluation from inner and outer surfaces  
- cross-section verification of aligned eroded and reconstructed surfaces  
- erosion-depth mapping relative to a reconstructed reference  
- geodesic erosion zoning from the intrados region  
- statistical analysis of erosion-depth distributions  
- export of figures and CSV data for further analysis  

The tools are intended for engineering, research, and educational use, particularly in erosion and wear studies.

---

## Repository Structure

```text
3d-eroded-elbow-scan-postprocessing/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── thickness_map.py
│   ├── cross_section_check.py
│   └── erosion_analysis_zones.py
│
├── data/
│   ├── cut_initial_thickness_1_7_mm.stl
│   ├── New_reconstructed_1_7.stl
│   └── New_reconstructed_aligned_1.stl
│
├── results/
│   ├── csv/
│   └── figures/
│
└── images/
```

---

## Included Scripts

### `thickness_map.py`

Interactive wall-thickness visualization for two-surface STL files.

**Main features:**
- automatically separates the two connected surfaces  
- identifies inner and outer walls  
- computes thickness using nearest-neighbor mapping  
- interactive point sampling (right-click)  
- UT-style grid generation on the outer surface  
- CSV export of sampled UT grid data  

---

### `cross_section_check.py`

Cross-section verification tool for aligned eroded and reconstructed surfaces.

**Main features:**
- separates connected components  
- estimates bend path using PCA  
- generates cross-sections normal to the bend path  
- compares eroded and reconstructed section profiles  

---

### `erosion_analysis_zones.py`

Full erosion-analysis workflow using eroded and reconstructed STL surfaces.

**Main features:**
- erosion depth computation  
- geodesic zoning from intrados  
- per-zone statistical analysis  
- distribution fitting (normal, lognormal, Weibull, etc.)  
- normality testing  
- bimodality detection (Zone 1)  
- CSV and figure export  .

---

## Input Data

The scripts operate on STL files stored in the `data/` folder.

Expected files:

- `cut_initial_thickness_1_7_mm.stl`  
- `New_reconstructed_1_7.stl`  
- `New_reconstructed_aligned_1.stl`  

You can also find there initial cleaned scan 

`3D_scan_clean.stl` 

If you decide to open it in Fusion 360 watch the scale. You can find correct scale in `images/` folder.
If the model is 10 times larger then apply 0.1 scale factor. 

---


## Results

Outputs are saved to:

- `results/csv/` — numerical data  
- `results/figures/` — plots and visualizations  

Typical outputs include:

- `zone_erosion_statistics.csv`  
- `zone1_bimodality_report.csv`  
- `erosion_depth_all_points.csv`  
- `erosion_depth_zone_*.csv`  
- `erosion_depth_zone_*_filtered.csv`  
- `ut_grid_thickness.csv`  

---

## Methods Used

The scripts combine geometric processing and statistical analysis techniques:

- connected-component separation  
- PCA-based geometric framing  
- nearest-neighbor search (KDTree)  
- point-to-surface erosion depth computation  
- geodesic zoning using graph shortest paths  
- distribution fitting (normal, exponential, gamma, lognormal, Weibull)  
- normality testing  
- KDE-based peak detection and bimodality analysis  

---

## Notes

- STL files must contain valid connected surfaces  
- Some workflows assume geometry is pre-aligned  
- Zone 0 represents near-zero erosion near the intrados and is excluded from some analyses by default  
- Scripts are standalone tools (not packaged as a Python library)  

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python scripts/thickness_map.py
python scripts/cross_section_check.py
python scripts/erosion_analysis_zones.py
```

---

## Author

Dr. Nikolay Bukharin

---

## License

MIT License
