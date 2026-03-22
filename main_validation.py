# main_validation.py

from prep import run_prep, convert_tif_to_png_renamed
from save_mask import run_sam2
from save_efd import run_efd_post_validation
from infer import run_mcdo_inference_all_frames


# =========================================================
# Configuration
# =========================================================
INDICES = list(range(1, 500))

RAW_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Imagesets/LD_exp"
ORIG_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_raw"
GAMMA_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_gamma"
SEG_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_seg"
EFD_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_efd"
RESULT_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_result_mcdo"

CAL_MODEL_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Calibration/LD_cal_model"

STATS_PATH = f"{CAL_MODEL_DIR}/stats_bbox.npy"
MODEL_PATH = f"{CAL_MODEL_DIR}/bnn_mcdo.pt"
TAU_PATH = f"{CAL_MODEL_DIR}/sigma_threshold_tau_bnn.npy"
FEATURE_MASK_PATH = f"{CAL_MODEL_DIR}/feature_mask.npy"
SCALER_PATH = f"{CAL_MODEL_DIR}/scaler_calib.joblib"

DEVICE = "cuda:1"


# =========================================================
# Pipeline switches
# =========================================================
RUN_CONVERT_RAW = True
RUN_GAMMA = True
RUN_SAM2 = True
RUN_EFD = True
RUN_INFER = True


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
    print("\n[STEP] validation EFD feature extraction")
    run_efd_post_validation(
        gamma_dir=GAMMA_DIR,
        raw_dir=ORIG_DIR,
        sam_dir=SEG_DIR,
        output_dir=EFD_DIR,
        indices=INDICES,
        stats_path=STATS_PATH,
    )


def step_infer() -> None:
    print("\n[STEP] MC-Dropout inference")
    run_mcdo_inference_all_frames(
        exp_dir=EFD_DIR,
        out_dir=RESULT_DIR,
        model_path=MODEL_PATH,
        tau_path=TAU_PATH,
        feature_mask_path=FEATURE_MASK_PATH,
        scaler_path=SCALER_PATH,
        indices=INDICES,
        use_harmonics=1,
        n_mc=200,
        batch=256,
        device="cuda",
    )


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("========================================")
    print(" Validation / Experiment Pipeline")
    print("========================================")

    if RUN_CONVERT_RAW:
        step_convert_raw()

    if RUN_GAMMA:
        step_gamma()

    if RUN_SAM2:
        step_sam2()

    if RUN_EFD:
        step_efd()

    if RUN_INFER:
        step_infer()

    print("\n[DONE] Validation pipeline finished.")


if __name__ == "__main__":
    main()