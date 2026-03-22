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

## Repository Structure

```text
.
├── main_calibration.py      # end-to-end calibration pipeline
├── main_validation.py       # end-to-end validation / experiment pipeline
├── prep.py                  # preprocessing: TIFF loading, normalization, gamma correction
├── save_mask.py             # SAM2 segmentation
├── save_efd.py              # EFD/intensity feature extraction
├── cal_ransac.py            # calibration feature filtering and scaler generation
├── mc_dropout.py            # MC-Dropout training on calibration data
├── infer.py                 # MC-Dropout inference on validation / experiment data
├── plot_result.py           # plotting utilities for inference results
└── README.md
