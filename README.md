# STAR-APTV

Deep learning-enabled 3D flow reconstruction in evaporating multicomponent droplets

This repository contains the core code used for the **STAR-APTV** pipeline proposed in:

**Bumsoo Park, Julius Mauch, Hyeokjin Kweon, Jochen Kriegseis, Seungchul Lee, Hyoungsoo Kim**  
*STAR-APTV: Deep learning-enabled 3D flow reconstruction in evaporating multicomponent droplets*  
**Measurement**, Volume 266, 2026, 120368.

---

## Overview

STAR-APTV is a deep learning-assisted astigmatic particle tracking velocimetry pipeline for robust 3D particle reconstruction from single-camera microscopic images.

The implementation in this repository focuses on the following stages:

1. **Raw image conversion and preprocessing**
   - TIFF to PNG conversion
   - gamma correction
   - grayscale normalization

2. **Particle segmentation**
   - SAM2-based automatic mask generation
   - phase-locked mask generation for improved spatial robustness

3. **Shape and intensity feature extraction**
   - elliptic Fourier descriptor (EFD)-based contour refinement
   - intensity-based feature extraction inside each refined particle mask
   - bounding-box-based filtering for validation/experiment data

4. **Calibration feature construction**
   - RANSAC-based filtering of calibration particles
   - feature masking and normalization
   - calibration statistics extraction

5. **Depth regression**
   - MC-Dropout-based regression model
   - predictive uncertainty estimation
   - uncertainty thresholding using calibration-derived tau

6. **Validation / experiment inference**
   - EFD feature extraction on experiment images
   - MC-Dropout inference across all frames
   - aggregated CSV export for downstream visualization and reconstruction

---

## Directory Structure

```text
.
├── Calibration/
│   ├── LD_cal_raw/          # renamed / converted calibration raw images
│   ├── LD_cal_gamma/        # gamma-corrected calibration images
│   ├── LD_cal_seg/          # SAM2 segmentation outputs for calibration
│   ├── LD_cal_efd/          # EFD/intensity feature outputs for calibration
│   └── LD_cal_model/        # calibration statistics, scaler, feature mask, MC-Dropout model, tau
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
├── .gitignore
└── README.md

---

## Pipeline

### 1. Calibration pipeline

The calibration pipeline performs:

- raw TIFF conversion
- gamma preprocessing
- SAM2 segmentation
- EFD/intensity feature extraction
- RANSAC-based calibration filtering
- MC-Dropout training

Run:

```bash
python main_calibration.py
```

### 2. Validation / experiment pipeline

The validation pipeline performs:

- raw TIFF conversion
- gamma preprocessing
- SAM2 segmentation
- EFD/intensity feature extraction using calibration-derived bbox statistics
- MC-Dropout inference using the calibration-trained model

Run:

```bash
python main_validation.py
```

---

## Data Convention

This code assumes indexed file naming for both calibration and validation data.

### Image naming

```text
0001.tif
0002.tif
0003.tif
...
```

### Intermediate outputs

```text
0001.png
Seg_0001.npy
EFD_Seg_0001.npy
```

The pipeline relies on consistent indexing across:
- raw images
- gamma-corrected images
- segmentation outputs
- EFD outputs

---

## Main Components

### Preprocessing
`prep.py`
- robust TIFF loading
- single-channel conversion
- max-based normalization to 8-bit
- gamma correction

### Segmentation
`save_mask.py`
- SAM2 automatic mask generation
- phase-locked grid shifting
- segmentation export as `.npy`

### EFD feature extraction
`save_efd.py`
- contour reconstruction via elliptic Fourier descriptors
- intensity feature extraction
- bbox estimation
- IoU-based duplicate suppression

### Calibration filtering
`cal_ransac.py`
- RANSAC-based feature filtering
- low-variance feature removal
- standardization and scaler export
- bbox statistics export

### MC-Dropout regression
`mc_dropout.py`
- MLP regressor with BatchNorm and Dropout
- uncertainty-aware regression
- tau threshold estimation from calibration uncertainty distribution

### Experiment inference
`infer.py`
- feature loading from experiment EFD outputs
- calibration-consistent masking and scaling
- MC-Dropout prediction with uncertainty
- CSV export of filtered predictions

---

## Output Files

Typical calibration outputs include:

```text
calibration_features.npy
calibration_features_normed.npy
scaler_calib.joblib
feature_mask.npy
stats_bbox.npy
bnn_mcdo.pt
sigma_threshold_tau_bnn.npy
```

Typical validation outputs include:

```text
EFD_Seg_0001.npy
EFD_Seg_0002.npy
...
all_exp_pred_mcdo.csv
```

---

## Dependencies

Typical Python dependencies include:

```text
numpy
scipy
matplotlib
opencv-python
scikit-image
scikit-learn
joblib
tifffile
pyefd
torch
Pillow
```

Install with:

```bash
pip install numpy scipy matplotlib opencv-python scikit-image scikit-learn joblib tifffile pyefd torch Pillow
```

---

## Notes

- Absolute paths are currently used in the scripts and should be adapted to your local environment.
- Large raw data, intermediate segmentation results, and trained model files are not intended to be version-controlled in GitHub by default.
- This repository is designed around the experimental and calibration file conventions used in the STAR-APTV study.

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
