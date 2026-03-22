import os
import joblib
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler


# ---------- 유틸: MAD ----------
def mad(x, axis=0):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(x - med), axis=axis)


# ---------- EFD/Intensity 벡터 만들기 ----------
def efd_feature_vec(output, use_harmonics=1):
    coeffs = output.get("EFD", None)
    if coeffs is None:
        return None
    H = min(use_harmonics, coeffs.shape[0])
    W = min(4 * use_harmonics, coeffs.shape[1])
    v = coeffs[:H, :W].reshape(-1)
    return v.astype(float)


INTENSITY_KEYS_ALL = [
    "int_mean",
    "int_std",
    "int_peak",
    "gauss_sigma_x",
    "gauss_sigma_y",
]


def intensity_feature_vec(output, keys=None):
    if keys is None:
        keys = INTENSITY_KEYS_ALL
    inten = output.get("intensity", {})
    return np.array([float(inten.get(k, np.nan)) for k in keys], dtype=float)


try:
    from statsmodels.robust.scale import mad as sm_mad

    def _mad(x, axis=0):
        return sm_mad(x, axis=axis)

except Exception:
    def _mad(x, axis=0):
        med = np.nanmedian(x, axis=axis, keepdims=True)
        return np.nanmedian(np.abs(x - med), axis=axis)


def ransac_inliers_per_feature(X_idx, Y, thr_scale=1.0, min_samples=0.3):
    N, D = Y.shape
    inlier_list = []

    thr = _mad(Y, axis=0) * float(thr_scale)
    thr_fallback = np.nanmedian(np.abs(Y - np.nanmedian(Y, axis=0)), axis=0) + 1e-6
    thr = np.where(np.isfinite(thr) & (thr > 0), thr, thr_fallback)

    for j in range(D):
        y = Y[:, j]

        if np.isnan(y).all():
            inlier_list.append(np.ones(N, dtype=bool))
            continue

        mask_valid = np.isfinite(y)
        if mask_valid.sum() < max(10, int(0.3 * N)):
            inlier_list.append(np.ones(N, dtype=bool))
            continue

        ransac = RANSACRegressor(
            residual_threshold=float(max(thr[j], 1e-6)),
            min_samples=min_samples,
            random_state=0,
        )
        ransac.fit(X_idx[mask_valid], y[mask_valid])

        in_mask = np.ones(N, dtype=bool)
        tmp = np.zeros(mask_valid.sum(), dtype=bool)
        tmp[:] = ransac.inlier_mask_
        in_mask[mask_valid] = tmp
        inlier_list.append(in_mask)

    return np.logical_and.reduce(inlier_list) if inlier_list else np.ones(N, dtype=bool)


def intensity_extreme_inliers(I, k_mad=3.0, min_valid_feats=2):
    I = I.astype(float)
    N, D = I.shape

    med = np.nanmedian(I, axis=0, keepdims=True)
    mads = mad(I, axis=0)
    mads = np.where(np.isfinite(mads) & (mads > 0), mads, 1e-6)

    lo = med - k_mad * mads
    hi = med + k_mad * mads

    finite = np.isfinite(I)
    in_range = (I >= lo) & (I <= hi) & finite

    valid_feats = in_range.sum(axis=1)
    ok = valid_feats >= min_valid_feats

    all_bad = (~finite).all(axis=1)
    ok = ok & (~all_bad)

    return ok


def stack_features_with_order(outputs, use_harmonics=1):
    efd_list = []
    maxd = 0
    for o in outputs:
        coeffs = o.get("EFD", None)
        if coeffs is None:
            efd_list.append(None)
            continue
        H = min(use_harmonics, coeffs.shape[0])
        W = min(4 * use_harmonics, coeffs.shape[1])
        v = coeffs[:H, :W].reshape(-1).astype(float)
        efd_list.append(v)
        maxd = max(maxd, v.size)

    E = np.full((len(outputs), maxd), np.nan)
    for i, v in enumerate(efd_list):
        if v is not None:
            E[i, :len(v)] = v

    def _intensity_feature_vec(out):
        inten = out.get("intensity", {})
        return np.array([
            float(inten.get("int_mean", np.nan)),
            float(inten.get("gauss_sigma_x", np.nan)),
            float(inten.get("gauss_sigma_y", np.nan)),
        ], dtype=float)

    I = np.vstack([_intensity_feature_vec(o) for o in outputs])
    return np.hstack([E, I])


def classify_by_ransac(
    outputs,
    use_harmonics=1,
    thr_scale_coeff=3.0,
    min_samples_coeff=0.3,
    k_mad_intensity=3.0,
    min_valid_feats_int=1,
    intensity_keys=None,
):
    N = len(outputs)
    X_idx = np.arange(N).reshape(-1, 1)

    efd_list = [efd_feature_vec(o, use_harmonics=use_harmonics) for o in outputs]
    dE = max((len(v) for v in efd_list if v is not None), default=0)

    E = np.full((N, dE), np.nan)
    for i, v in enumerate(efd_list):
        if v is not None:
            E[i, :len(v)] = v

    if intensity_keys is None:
        keys = INTENSITY_KEYS_ALL
    else:
        keys = list(intensity_keys)

    I = np.vstack([intensity_feature_vec(o, keys=keys) for o in outputs])
    EI = np.hstack([E, I])

    coeff_in = (
        ransac_inliers_per_feature(
            X_idx,
            EI,
            thr_scale=thr_scale_coeff,
            min_samples=min_samples_coeff,
        )
        if dE > 0
        else np.ones(N, dtype=bool)
    )

    valid = coeff_in
    reasons = []
    for i in range(N):
        r = []
        if not coeff_in[i]:
            r.append("coeff_outlier")
        reasons.append("; ".join(r) if r else "ok")

    return {
        "coeff_inlier": coeff_in,
        "valid": valid,
        "reasons": reasons,
    }


def build_and_save_calibration(
    efd_dir,
    frames,
    out_path,
    scaler_path,
    normed_out_path,
    use_harmonics=1,
    thr_scale_coeff=3.0,
    min_samples_coeff=0.3,
    k_mad_intensity=2.0,
    min_valid_feats_int=2,
    low_variance_threshold=1e-10,
    intensity_keys=("int_mean", "gauss_sigma_x", "gauss_sigma_y"),
):
    rows_all = []
    total_seen, total_kept = 0, 0
    widths_all, heights_all, ratios_all = [], [], []

    for i in frames:
        npy_path = os.path.join(efd_dir, f"EFD_Seg_{i:04d}.npy")
        if not os.path.exists(npy_path):
            print(f"[skip] {npy_path} not found")
            continue

        outputs = np.load(npy_path, allow_pickle=True)
        total_seen += len(outputs)

        cls = classify_by_ransac(
            outputs,
            use_harmonics=use_harmonics,
            thr_scale_coeff=thr_scale_coeff,
            min_samples_coeff=min_samples_coeff,
            k_mad_intensity=k_mad_intensity,
            min_valid_feats_int=min_valid_feats_int,
            intensity_keys=intensity_keys,
        )

        keep = np.asarray(cls["valid"]).astype(bool)
        if keep.sum() == 0:
            print(f"[OK] frame {i}: loaded {len(outputs)} → kept 0")
            continue

        kept_outputs = [o for o, k in zip(outputs, keep) if k]

        for o in kept_outputs:
            bx, by, bw, bh = o.get("bbox", (np.nan, np.nan, np.nan, np.nan))
            if np.isfinite(bw) and np.isfinite(bh) and bh > 0:
                widths_all.append(float(bw))
                heights_all.append(float(bh))
                ratios_all.append(float(bw) / float(bh))

        X = stack_features_with_order(kept_outputs, use_harmonics=use_harmonics)
        mask = ~np.all(np.isnan(X), axis=0)
        X = X[:, mask]

        z_col = np.full((X.shape[0], 1), float(i))
        rows = np.hstack([X, z_col])
        rows_all.append(rows)

        total_kept += X.shape[0]
        print(f"[OK] frame {i}: loaded {len(outputs)} → kept {X.shape[0]}")

    if not rows_all:
        raise RuntimeError("No samples kept after RANSAC filtering.")

    data = np.vstack(rows_all)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, data)
    print(f"[SAVED] {out_path} shape={data.shape} (kept {total_kept}/{total_seen}, {total_kept/total_seen:.1%})")

    X = data[:, :-1]
    z = data[:, -1:]

    row_mask = np.all(np.isfinite(X), axis=1)
    X_for_var = X[row_mask]

    vt = VarianceThreshold(threshold=low_variance_threshold)
    vt.fit(X_for_var)
    feature_mask = vt.get_support()
    kept_idx = np.where(feature_mask)[0]
    print(
        f"[FEAT] low-variance removal: kept={kept_idx.size}, "
        f"dropped={X.shape[1] - kept_idx.size} (thr={low_variance_threshold})"
    )

    X_kept = X[:, feature_mask]

    row_mask2 = np.all(np.isfinite(X_kept), axis=1)
    X_fit = X_kept[row_mask2]
    scaler = StandardScaler().fit(X_fit)
    joblib.dump(scaler, scaler_path)
    print(f"[SAVED] Scaler → {scaler_path} (fit on {X_fit.shape[0]} rows, D_after={X_kept.shape[1]})")

    X_norm = scaler.transform(X_kept)
    data_normed = np.hstack([X_norm, z])
    np.save(normed_out_path, data_normed)
    print(f"[SAVED] Normalized data → {normed_out_path} shape={data_normed.shape}")

    np.save(os.path.join(os.path.dirname(normed_out_path), "feature_mask.npy"), feature_mask.astype(np.uint8))
    np.save(os.path.join(os.path.dirname(normed_out_path), "feature_indices_kept.npy"), kept_idx.astype(np.int32))

    if len(widths_all) > 0:
        stats = {
            "min_width": float(np.min(widths_all)),
            "max_width": float(np.max(widths_all)),
            "min_height": float(np.min(heights_all)),
            "max_height": float(np.max(heights_all)),
            "min_aspect_ratio": float(np.min(ratios_all)),
            "max_aspect_ratio": float(np.max(ratios_all)),
        }
        stats_path = os.path.join(os.path.dirname(normed_out_path), "stats_bbox.npy")
        np.save(stats_path, stats)
        print("[SAVED] bbox stats →", stats_path)
        for k, v in stats.items():
            print(f"   {k}: {v:.3f}")
    else:
        print("[WARN] No bbox data found for stats.")

    return data, data_normed, scaler


def run_calibration_ransac(
    efd_dir,
    indices,
    model_dir,
    use_harmonics=1,
    thr_scale_coeff=2.0,
    min_samples_coeff=0.3,
    k_mad_intensity=2.0,
    min_valid_feats_int=2,
    low_variance_threshold=1e-10,
    intensity_keys=("int_mean", "gauss_sigma_x", "gauss_sigma_y"),
):
    os.makedirs(model_dir, exist_ok=True)

    out_path = os.path.join(model_dir, "calibration_features.npy")
    scaler_path = os.path.join(model_dir, "scaler_calib.joblib")
    normed_out_path = os.path.join(model_dir, "calibration_features_normed.npy")

    return build_and_save_calibration(
        efd_dir=efd_dir,
        frames=indices,
        out_path=out_path,
        scaler_path=scaler_path,
        normed_out_path=normed_out_path,
        use_harmonics=use_harmonics,
        thr_scale_coeff=thr_scale_coeff,
        min_samples_coeff=min_samples_coeff,
        k_mad_intensity=k_mad_intensity,
        min_valid_feats_int=min_valid_feats_int,
        low_variance_threshold=low_variance_threshold,
        intensity_keys=intensity_keys,
    )