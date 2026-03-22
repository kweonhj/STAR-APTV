import os
from time import time

import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors, reconstruct_contour
from scipy.ndimage import binary_closing
from scipy.optimize import curve_fit
from skimage.measure import approximate_polygon
from skimage.morphology import remove_small_objects


def _gaussian2d_mesh(xy_tuple, A, x0, y0, sx, sy, theta, offset):
    (X, Y) = xy_tuple
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xc = cos_t * (X - x0) + sin_t * (Y - y0)
    Yc = -sin_t * (X - x0) + cos_t * (Y - y0)
    g = A * np.exp(-0.5 * ((Xc / sx) ** 2 + (Yc / sy) ** 2)) + offset
    return g.ravel()


def intensity_centroid(image, mask):
    vals = image * mask
    S = vals.sum()
    if S <= 0:
        return {"x_c": np.nan, "y_c": np.nan}
    yy, xx = np.indices(image.shape)
    x_c = float((xx * vals).sum() / S)
    y_c = float((yy * vals).sum() / S)
    return {"x_c": x_c, "y_c": y_c}


def fit_gaussian2d(image, mask, min_size=20):
    m = mask.astype(bool)
    m = remove_small_objects(m, min_size=min_size)
    m = binary_closing(m, structure=np.ones((3, 3), bool))
    if m.sum() == 0:
        return {"success": False}

    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    roi = image[y0:y1 + 1, x0:x1 + 1]
    mroi = m[y0:y1 + 1, x0:x1 + 1]

    yy, xx = np.indices(roi.shape)
    vals = roi * mroi
    S = vals.sum()
    if S <= 0:
        return {"success": False}

    x_c = (xx * vals).sum() / S
    y_c = (yy * vals).sum() / S
    x2 = (((xx - x_c) ** 2) * vals).sum() / S
    y2 = (((yy - y_c) ** 2) * vals).sum() / S
    sx0 = np.sqrt(max(x2, 1e-6))
    sy0 = np.sqrt(max(y2, 1e-6))

    A0 = vals.max() - np.median(vals[mroi])
    offset0 = np.median(vals[mroi])

    X, Y = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
    p0 = [float(A0), float(x_c), float(y_c), float(sx0), float(sy0), 0.0, float(offset0)]
    bounds = (
        [0.0, 0.0, 0.0, 0.5, 0.5, -np.pi / 2, -np.inf],
        [np.inf, roi.shape[1], roi.shape[0], 50.0, 50.0, np.pi / 2, np.inf]
    )

    try:
        popt, _ = curve_fit(
            _gaussian2d_mesh, (X, Y), roi.ravel(),
            p0=p0, bounds=bounds, maxfev=2000
        )
        A, x0f, y0f, sxf, syf, thetaf, off = popt
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        return {
            "success": True,
            "A": float(A),
            "x0": float(x0f + x0),
            "y0": float(y0f + y0),
            "sx": float(sxf),
            "sy": float(syf),
            "theta": float(thetaf),
            "offset": float(off),
            "fwhm_x": float(fwhm * sxf),
            "fwhm_y": float(fwhm * syf)
        }
    except Exception:
        c = intensity_centroid(roi, mroi)
        return {
            "success": False,
            "x0": float(c["x_c"] + x0),
            "y0": float(c["y_c"] + y0)
        }


def radial_intensity_profile(image, mask, n_rays=64):
    if mask.sum() == 0:
        return np.zeros(n_rays, dtype=float)

    c = intensity_centroid(image, mask)
    cy, cx = c["y_c"], c["x_c"]
    if np.isnan(cx) or np.isnan(cy):
        cy, cx = np.mean(np.where(mask), axis=1)

    H, W = image.shape
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    prof = []
    for th in angles:
        dx, dy = np.cos(th), np.sin(th)
        vals = []
        for r in range(1, 9999):
            x = int(round(cx + r * dx))
            y = int(round(cy + r * dy))
            if x < 0 or x >= W or y < 0 or y >= H:
                break
            if not mask[y, x]:
                break
            vals.append(image[y, x])
        prof.append(np.mean(vals) if len(vals) else 0.0)
    return np.array(prof, dtype=float)


def extract_intensity_features(image_gray, mask_bool):
    img = image_gray.astype(np.float32)
    if img.max() > 2.0:
        img = img / 255.0

    m = mask_bool.astype(bool)
    vals = img[m]

    if vals.size == 0:
        return {
            "int_mean": 0.0,
            "int_std": 0.0,
            "int_peak": 0.0,
            "gauss_sigma_x": np.nan,
            "gauss_sigma_y": np.nan,
        }

    int_mean = float(vals.mean())
    int_std = float(vals.std())
    int_peak = float(vals.max())

    ys, xs = np.where(m)
    w = vals.astype(np.float64)
    w_sum = w.sum()

    if w_sum <= 0:
        sigma_x = np.nan
        sigma_y = np.nan
    else:
        mx = float((xs * w).sum() / w_sum)
        my = float((ys * w).sum() / w_sum)
        var_x = float(((xs - mx) ** 2 * w).sum() / w_sum)
        var_y = float(((ys - my) ** 2 * w).sum() / w_sum)
        sigma_x = np.sqrt(max(var_x, 0.0))
        sigma_y = np.sqrt(max(var_y, 0.0))

    return {
        "int_mean": int_mean,
        "gauss_sigma_x": sigma_x,
        "gauss_sigma_y": sigma_y
    }


def mask_to_efd(binary_mask, order=10, prev_theta=None, prev_phi=None):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3))).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found in the binary mask.")
        return binary_mask, None

    largest_contour = max(contours, key=cv2.contourArea)
    contour_array = largest_contour[:, 0, :]

    def remove_translation(contour):
        centroid = np.mean(contour, axis=0)
        return contour - centroid

    def resample_contour(contour, num_points=100):
        arc_length = cv2.arcLength(contour, closed=True)
        approx_contour = approximate_polygon(contour, tolerance=arc_length / num_points)
        return approx_contour

    zero_mean_contour = remove_translation(resample_contour(contour_array))
    locus = np.mean(contour_array, axis=0)

    coeffs = elliptic_fourier_descriptors(zero_mean_contour, order=order, normalize=False)

    A1, B1, C1, D1 = coeffs[0]
    theta = 0.5 * np.arctan2(
        2 * (A1 * B1 + C1 * D1),
        (A1 ** 2 - B1 ** 2 + C1 ** 2 - D1 ** 2)
    )

    if prev_theta is not None:
        delta = (theta - prev_theta + np.pi) % (2 * np.pi) - np.pi
        theta = prev_theta + delta

    for n in range(1, coeffs.shape[0] + 1):
        block = np.array([[coeffs[n - 1, 0], coeffs[n - 1, 1]],
                          [coeffs[n - 1, 2], coeffs[n - 1, 3]]])
        R = np.array([[np.cos(n * theta), -np.sin(n * theta)],
                      [np.sin(n * theta), np.cos(n * theta)]])
        coeffs[n - 1, :] = (block @ R).flatten()

    A1r, B1r, C1r, D1r = coeffs[0]
    phi = np.arctan2(B1r, A1r)
    if prev_phi is not None:
        dphi = (phi - prev_phi + np.pi) % (2 * np.pi) - np.pi
        phi = prev_phi + dphi

    def apply_phase(block, n):
        R = np.array([[np.cos(n * phi), -np.sin(n * phi)],
                      [np.sin(n * phi), np.cos(n * phi)]])
        return block @ R

    for n in range(1, coeffs.shape[0] + 1):
        block = np.array([[coeffs[n - 1, 0], coeffs[n - 1, 1]],
                          [coeffs[n - 1, 2], coeffs[n - 1, 3]]])
        coeffs[n - 1, :] = apply_phase(block, n).flatten()

    if coeffs[0, 0] < 0:
        coeffs *= -1.0

    contour_recon = reconstruct_contour(coeffs, locus=locus, num_points=len(contour_array))
    contour_recon = np.array(contour_recon, dtype=np.int32).reshape((-1, 1, 2))
    reconstructed_mask = np.zeros_like(binary_mask)
    cv2.drawContours(reconstructed_mask, [contour_recon], -1, 255, thickness=-1)

    return reconstructed_mask, coeffs


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def suppress_by_iou_keep_smaller(outputs, iou_thr=0.9):
    N = len(outputs)
    keep = [True] * N
    for i in range(N):
        if not keep[i]:
            continue
        mi = outputs[i]["segmentation"].astype(bool)
        ai = int(outputs[i].get("area", mi.sum()))
        for j in range(i + 1, N):
            if not keep[j]:
                continue
            mj = outputs[j]["segmentation"].astype(bool)
            aj = int(outputs[j].get("area", mj.sum()))
            iou = mask_iou(mi, mj)
            if iou >= iou_thr:
                if ai <= aj:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [outputs[k] for k in range(N) if keep[k]]


def _bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if ys.size > 0:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        w = int(x1 - x0 + 1)
        h = int(y1 - y0 + 1)
        return (x0, y0, w, h)
    return (np.nan, np.nan, np.nan, np.nan)


def _load_bbox_stats(stats_path):
    bbox_stats = np.load(stats_path, allow_pickle=True).item()
    return {
        "MIN_W": float(bbox_stats["min_width"]),
        "MAX_W": float(bbox_stats["max_width"]),
        "MIN_H": float(bbox_stats["min_height"]),
        "MAX_H": float(bbox_stats["max_height"]),
        "MIN_AR": float(bbox_stats["min_aspect_ratio"]),
        "MAX_AR": float(bbox_stats["max_aspect_ratio"]),
    }


def _bbox_passes_stats(mask_bool, bbox_stats):
    bbox = _bbox_from_mask(mask_bool)
    x0, y0, w, h = bbox

    if not np.isfinite(w) or not np.isfinite(h) or h <= 0:
        return False, bbox

    aspect_ratio = float(w) / float(h)

    if (w < bbox_stats["MIN_W"]) or (w > bbox_stats["MAX_W"]):
        return False, bbox
    if (h < bbox_stats["MIN_H"]) or (h > bbox_stats["MAX_H"]):
        return False, bbox
    if (aspect_ratio < bbox_stats["MIN_AR"]) or (aspect_ratio > bbox_stats["MAX_AR"]):
        return False, bbox

    return True, bbox


def _process_single_output(output, img_original, bbox_stats=None):
    bool_mask_in = output["segmentation"].astype(bool)

    print("mask shape:", bool_mask_in.shape)
    print("mask True pixels:", bool_mask_in.sum())

    prev_area = int(bool_mask_in.sum())
    if prev_area < 10 or prev_area > 1000:
        return None

    reconstructed_mask, efd_coeff = mask_to_efd(bool_mask_in, order=10)

    bool_mask = reconstructed_mask.astype(bool)
    new_area = int(bool_mask.sum())
    if new_area > 1000:
        return None

    if bbox_stats is not None:
        ok_bbox, bbox = _bbox_passes_stats(bool_mask, bbox_stats)
        if not ok_bbox:
            return None
    else:
        bbox = _bbox_from_mask(bool_mask)

    intensity_feat = extract_intensity_features(img_original, bool_mask)

    output["segmentation"] = bool_mask
    output["EFD"] = efd_coeff
    output["area"] = new_area
    output["intensity"] = intensity_feat
    output["bbox"] = bbox

    print(intensity_feat)
    return output


def run_efd_post_calibration(gamma_dir, raw_dir, sam_dir, output_dir, indices):
    os.makedirs(output_dir, exist_ok=True)

    for index in indices:
        image_path = os.path.join(gamma_dir, f"{index:04d}.png")
        original_path = os.path.join(raw_dir, f"{index:04d}.png")
        sam_path = os.path.join(sam_dir, f"Seg_{index:04d}.npy")
        save_path = os.path.join(output_dir, f"EFD_Seg_{index:04d}.npy")

        if not os.path.exists(sam_path):
            print(f"File not found: {sam_path}")
            continue
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Failed to load: {image_path}")
            continue

        img_original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            print(f"Failed to load: {original_path}")
            continue

        sam_outputs = np.load(sam_path, allow_pickle=True)
        filtered_outputs = []

        start_time = time()
        for output in sam_outputs:
            processed = _process_single_output(output, img_original, bbox_stats=None)
            if processed is not None:
                filtered_outputs.append(processed)

        end_time = time()
        print("Time: ", end_time - start_time, "s")

        start_time = time()
        filtered_outputs = suppress_by_iou_keep_smaller(filtered_outputs, iou_thr=0.2)
        end_time = time()
        print("Time: ", end_time - start_time, "s")

        if filtered_outputs:
            np.save(save_path, filtered_outputs)
            print(f"Processed and saved: {save_path}  (kept {len(filtered_outputs)} segments)")
        else:
            print(f"All segments dropped for: {save_path} (No valid areas ≤ 1000)")


def run_efd_post_validation(
    gamma_dir,
    raw_dir,
    sam_dir,
    output_dir,
    indices,
    stats_path,
):
    os.makedirs(output_dir, exist_ok=True)
    bbox_stats = _load_bbox_stats(stats_path)

    print("[BBOX STATS]")
    print(" min_width , max_width :", bbox_stats["MIN_W"], bbox_stats["MAX_W"])
    print(" min_height, max_height:", bbox_stats["MIN_H"], bbox_stats["MAX_H"])
    print(" min_ar    , max_ar    :", bbox_stats["MIN_AR"], bbox_stats["MAX_AR"])

    for index in indices:
        idx = f"{int(index):04d}"

        image_path = os.path.join(gamma_dir, f"{idx}.png")
        original_path = os.path.join(raw_dir, f"{idx}.png")
        sam_path = os.path.join(sam_dir, f"Seg_{idx}.npy")
        save_path = os.path.join(output_dir, f"EFD_Seg_{idx}.npy")

        if not os.path.exists(sam_path):
            print(f"File not found: {sam_path}")
            continue
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Failed to load: {image_path}")
            continue

        if not os.path.exists(original_path):
            raise FileNotFoundError(f"RAW PNG not found: {original_path}")

        img_original = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
        if img_original is None:
            print(f"Failed to load: {original_path}")
            continue

        if img_original.ndim == 3:
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        img_original = img_original.astype(np.float32)
        img_original /= 65535.0

        sam_outputs = np.load(sam_path, allow_pickle=True)
        filtered_outputs = []

        start_time = time()
        for output in sam_outputs:
            processed = _process_single_output(output, img_original, bbox_stats=bbox_stats)
            if processed is not None:
                filtered_outputs.append(processed)

        end_time = time()
        print("Time: ", end_time - start_time, "s")

        start_time = time()
        filtered_outputs = suppress_by_iou_keep_smaller(filtered_outputs, iou_thr=0.2)
        end_time = time()
        print("Time: ", end_time - start_time, "s")

        if filtered_outputs:
            np.save(save_path, filtered_outputs)
            print(f"Processed and saved: {save_path}  (kept {len(filtered_outputs)} segments)")
        else:
            print(f"All segments dropped for: {save_path} (No valid areas ≤ 1000)")