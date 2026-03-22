# main_calibration.py

from prep import run_prep, convert_tif_to_png_renamed
from save_mask import run_sam2
from save_efd import run_efd_post_calibration
from cal_ransac import run_calibration_ransac
from mc_dropout import run_mc_dropout


# =========================================================
# Configuration
# =========================================================
INDICES = list(range(1, 40))

RAW_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Imagesets/LD_cal"
ORIG_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Calibration/LD_cal_raw"
GAMMA_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Calibration/LD_cal_gamma"
SEG_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Calibration/LD_cal_seg"
EFD_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Calibration/LD_cal_efd"
MODEL_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Calibration/LD_cal_model"

DEVICE = "cuda:1"


# =========================================================
# Pipeline switches
# =========================================================
RUN_CONVERT_RAW = True
RUN_GAMMA = False
RUN_SAM2 = False
RUN_EFD = False
RUN_RANSAC = False
RUN_MCDO = False


# =========================================================
# Step functions
# =========================================================
def step_convert_raw() -> None:
    print("\n[STEP] convert raw tif -> renamed png")
    convert_tif_to_png_renamed(
        input_dir=RAW_DIR,
        output_dir=ORIG_DIR,
    )


def step_gamma() -> None:
    print("\n[STEP] gamma preprocessing")
    run_prep(
        input_dir=ORIG_DIR,
        output_dir=GAMMA_DIR,
        gamma_value=1.0,
        keep_stem=False,
    )


def step_sam2() -> None:
    print("\n[STEP] SAM2 segmentation")
    run_sam2(
        input_dir=GAMMA_DIR,
        output_dir=SEG_DIR,
        indices=INDICES,
        device_str=DEVICE,
    )


def step_efd() -> None:
    print("\n[STEP] EFD feature extraction")
    run_efd_post_calibration(
        gamma_dir=GAMMA_DIR,
        raw_dir=ORIG_DIR,
        sam_dir=SEG_DIR,
        output_dir=EFD_DIR,
        indices=INDICES,
    )


def step_ransac() -> None:
    print("\n[STEP] calibration RANSAC filtering + feature bank")
    run_calibration_ransac(
        efd_dir=EFD_DIR,
        indices=INDICES,
        model_dir=MODEL_DIR,
        use_harmonics=1,
        thr_scale_coeff=2.0,
        min_samples_coeff=0.3,
        k_mad_intensity=2.0,
        min_valid_feats_int=2,
        low_variance_threshold=1e-10,
        intensity_keys=["int_mean", "gauss_sigma_x", "gauss_sigma_y"],
    )


def step_mcdo() -> None:
    print("\n[STEP] MC-Dropout training")
    run_mc_dropout(
        calib_path=f"{MODEL_DIR}/calibration_features_normed.npy",
        model_path=f"{MODEL_DIR}/bnn_mcdo.pt",
        tau_path=f"{MODEL_DIR}/sigma_threshold_tau_bnn.npy",
        seed=42,
        test_size=0.2,
        random_state=0,
        batch=128,
        epochs=2000,
        lr=5e-4,
        wd=1e-5,
        dropout_p=0.2,
        hidden=(384, 384),
        use_huber=False,
        huber_delta=1.0,
        n_mc=200,
        patience=200,
        use_sig_cal=True,
        cover_target=0.95,
        device=None,
    )


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("========================================")
    print(" Calibration Pipeline")
    print("========================================")

    if RUN_CONVERT_RAW:
        step_convert_raw()

    if RUN_GAMMA:
        step_gamma()

    if RUN_SAM2:
        step_sam2()

    if RUN_EFD:
        step_efd()

    if RUN_RANSAC:
        step_ransac()

    if RUN_MCDO:
        step_mcdo()

    print("\n[DONE] Calibration pipeline finished.")


if __name__ == "__main__":
    main()