# STAR-APTV

Deep learning-enabled 3D flow reconstruction in evaporating multicomponent droplets

This repository contains the code used for the **STAR-APTV** pipeline proposed in:

**Bumsoo Park, Julius Mauch, Hyeokjin Kweon, Jochen Kriegseis, Seungchul Lee, Hyoungsoo Kim**  
*STAR-APTV: Deep learning-enabled 3D flow reconstruction in evaporating multicomponent droplets*  
**Measurement**, Volume 266, 2026, 120368.

---

## Overview

STAR-APTV is a deep learning-assisted astigmatic particle tracking velocimetry pipeline for robust 3D particle reconstruction from single-camera microscopic images.

This repository implements the following stages:

1. Raw image conversion and preprocessing
   - TIFF to PNG conversion
   - grayscale normalization
   - gamma correction

2. Particle segmentation
   - SAM2-based automatic mask generation
   - phase-locked mask generation for improved robustness

3. Shape and intensity feature extraction
   - elliptic Fourier descriptor (EFD)-based contour refinement
   - intensity-based feature extraction
   - bounding-box-based filtering for validation / experiment data

4. Calibration feature construction
   - RANSAC-based calibration filtering
   - low-variance feature removal
   - feature masking and normalization
   - calibration statistics extraction

5. Depth regression
   - MC-Dropout-based regression model
   - predictive uncertainty estimation
   - uncertainty thresholding using calibration-derived tau

6. Validation / experiment inference
   - EFD feature extraction on experiment images
   - MC-Dropout inference across all frames
   - CSV export for downstream reconstruction and visualization

---

## Repository Structure

```text
.
├── Calibration/
│   ├── LD_cal_raw/          # renamed / converted calibration raw images
│   ├── LD_cal_gamma/        # gamma-corrected calibration images
│   ├── LD_cal_seg/          # SAM2 segmentation outputs for calibration
│   ├── LD_cal_efd/          # EFD/intensity feature outputs for calibration
│   └── LD_cal_model/        # calibration statistics, scaler, feature mask, trained model, tau
│
├── Experiment/
│   ├── LD_exp_raw/          # renamed / converted experiment raw images
│   ├── LD_exp_gamma/        # gamma-corrected experiment images
│   ├── LD_exp_seg/          # SAM2 segmentation outputs for experiment data
│   ├── LD_exp_efd/          # EFD/intensity feature outputs for experiment data
│   └── LD_exp_result_mcdo/  # MC-Dropout inference results
│
├── Imagesets/
│   ├── LD_cal/              # original calibration TIFF images
│   └── LD_exp/              # original experiment TIFF images
│
├── main_calibration.py
├── main_validation.py
├── prep.py
├── save_mask.py
├── save_efd.py
├── cal_ransac.py
├── mc_dropout.py
├── infer.py
├── plot_result.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Organization

The repository separates original raw images from generated pipeline outputs.

- `Imagesets/` contains the original calibration and experiment TIFF images.
- `Calibration/` contains all generated outputs derived from calibration raw images.
- `Experiment/` contains all generated outputs derived from experiment raw images.

In other words, the raw source images are stored under `Imagesets/`, while all intermediate and final outputs produced by the code are written into either `Calibration/` or `Experiment/`.

---

## File Naming Convention

The code assumes indexed file naming.

### Raw images

```text
0001.tif
0002.tif
0003.tif
...
```

### Converted / processed images

```text
0001.png
0002.png
...
```

### Intermediate outputs

```text
Seg_0001.npy
Seg_0002.npy
EFD_Seg_0001.npy
EFD_Seg_0002.npy
...
```

The pipeline relies on consistent indexing across:
- raw images
- gamma-corrected images
- segmentation outputs
- EFD outputs

---

## Main Scripts

### `main_calibration.py`
Runs the full calibration pipeline:
- TIFF conversion
- gamma preprocessing
- SAM2 segmentation
- EFD/intensity feature extraction
- RANSAC-based calibration feature generation
- MC-Dropout training

### `main_validation.py`
Runs the full validation / experiment pipeline:
- TIFF conversion
- gamma preprocessing
- SAM2 segmentation
- EFD/intensity feature extraction with calibration-derived bbox statistics
- MC-Dropout inference on experiment frames

---

## Core Modules

### `prep.py`
Preprocessing utilities:
- robust TIFF loading
- single-channel conversion
- max-based normalization to 8-bit
- gamma correction

### `save_mask.py`
SAM2 segmentation utilities:
- automatic mask generation
- phase-locked grid shifting
- segmentation export as `.npy`

### `save_efd.py`
Shape and intensity feature extraction:
- contour refinement via elliptic Fourier descriptors
- intensity feature extraction
- bbox estimation
- IoU-based duplicate suppression
- separate routines for calibration and validation / experiment data

### `cal_ransac.py`
Calibration feature processing:
- RANSAC-based filtering
- low-variance feature removal
- scaler generation
- feature mask generation
- bbox statistics export

### `mc_dropout.py`
Calibration-time regression training:
- MLP regressor with BatchNorm and Dropout
- uncertainty-aware MC-Dropout regression
- tau threshold estimation from calibration uncertainty

### `infer.py`
Validation-time inference:
- feature loading from experiment EFD outputs
- calibration-consistent masking and scaling
- MC-Dropout prediction with uncertainty
- aggregated CSV export

### `plot_result.py`
Visualization utilities for experiment inference results.

---

## Pipeline

### 1. Calibration pipeline

The calibration pipeline performs:

- conversion of raw TIFF images
- gamma preprocessing
- SAM2 segmentation
- EFD/intensity feature extraction
- RANSAC-based calibration feature filtering
- MC-Dropout training

Run:

```bash
python main_calibration.py
```

### 2. Validation / experiment pipeline

The validation pipeline performs:

- conversion of raw TIFF images
- gamma preprocessing
- SAM2 segmentation
- EFD/intensity feature extraction using calibration-derived bbox statistics
- MC-Dropout inference using the calibration-trained model

Run:

```bash
python main_validation.py
```

---

## Typical Outputs

### Calibration outputs

Typical files produced under `Calibration/LD_cal_model/` include:

```text
calibration_features.npy
calibration_features_normed.npy
scaler_calib.joblib
feature_mask.npy
stats_bbox.npy
bnn_mcdo.pt
sigma_threshold_tau_bnn.npy
```

### Experiment outputs

Typical files produced under `Experiment/LD_exp_result_mcdo/` include:

```text
all_exp_pred_mcdo.csv
plot_3d_scatter.png
plot_top_view_xy_zcolor.png
plot_side_view_xz.png
plot_uncertainty_hist.png
plot_framewise_summary.png
```

---

## Dependencies

Tested environment:

```text
numpy==2.2.6
scipy==1.15.3
matplotlib==3.10.6
opencv-python==4.12.0.88
scikit-image==0.25.2
scikit-learn==1.7.2
joblib==1.5.2
tifffile==2025.5.10
pyefd==1.6.0
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
Pillow==11.3.0
```

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

For SAM2, install the local package separately from your cloned SAM2 repository, for example:

```bash
cd /path/to/sam2
pip install -e .
```

---

## Notes

- Absolute paths are currently used in the scripts and should be adapted to your local environment.
- Large raw data, intermediate segmentation results, and trained model files are not intended to be version-controlled by default.
- The repository is organized so that original TIFF images remain under `Imagesets/`, while generated outputs are stored under `Calibration/` and `Experiment/`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{PARK2026120368,
  title   = {STAR-APTV: Deep learning-enabled 3D flow reconstruction in evaporating multicomponent droplets},
  journal = {Measurement},
  volume  = {266},
  pages   = {120368},
  year    = {2026},
  author  = {Bumsoo Park and Julius Mauch and Hyeokjin Kweon and Jochen Kriegseis and Seungchul Lee and Hyoungsoo Kim}
}
```

---

## Contact

For questions regarding the code or implementation details, please contact the authors of the paper.
