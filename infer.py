import os
import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


class MLP_MCDropout(nn.Module):
    def __init__(self, in_dim, hidden, p=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p),
            ]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def efd_feature_vec(output, use_harmonics=1):
    coeffs = output.get("EFD", None)
    if coeffs is None:
        return None
    H = min(use_harmonics, coeffs.shape[0])
    W = min(4 * use_harmonics, coeffs.shape[1])
    return coeffs[:H, :W].reshape(-1).astype(float)


def intensity_feature_vec(output):
    inten = output.get("intensity", {})
    return np.array(
        [
            float(inten.get("int_mean", np.nan)),
            float(inten.get("gauss_sigma_x", np.nan)),
            float(inten.get("gauss_sigma_y", np.nan)),
        ],
        dtype=float,
    )


def stack_features_with_order(outputs, use_harmonics=1):
    efd_list = []
    dmax = 0
    for o in outputs:
        v = efd_feature_vec(o, use_harmonics=use_harmonics)
        efd_list.append(v)
        if v is not None:
            dmax = max(dmax, v.size)

    E = np.full((len(outputs), dmax), np.nan)
    for j, v in enumerate(efd_list):
        if v is not None:
            E[j, :len(v)] = v

    I = np.vstack([intensity_feature_vec(o) for o in outputs])
    return np.hstack([E, I])


def extract_geom(outputs, frame_idx_val):
    frame_idx = []
    x_centers = []
    y_centers = []
    bbox_w = []
    bbox_h = []

    for o in outputs:
        bx, by, bw, bh = o.get("bbox", (np.nan, np.nan, np.nan, np.nan))
        bx = float(bx)
        by = float(by)
        bw = float(bw)
        bh = float(bh)

        xc = bx + bw / 2.0
        yc = by + bh / 2.0

        frame_idx.append(int(frame_idx_val))
        x_centers.append(xc)
        y_centers.append(yc)
        bbox_w.append(bw)
        bbox_h.append(bh)

    return (
        np.array(frame_idx, dtype=int),
        np.array(x_centers, dtype=float),
        np.array(y_centers, dtype=float),
        np.array(bbox_w, dtype=float),
        np.array(bbox_h, dtype=float),
    )


def load_checkpoint(model_path: str, device="cuda"):
    ckpt = torch.load(model_path, map_location=device)
    in_dim = int(ckpt["in_dim"])
    hidden = list(map(int, ckpt["hidden"]))
    dropout_p = float(ckpt["dropout_p"])
    y_mu = float(ckpt["y_mu"])
    y_sd = float(ckpt["y_sd"])
    sigma_scale = float(ckpt.get("sigma_scale", 1.0))

    model = MLP_MCDropout(in_dim, hidden, p=dropout_p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, y_mu, y_sd, sigma_scale, in_dim


def enable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_mc(model, X: np.ndarray, n_mc=200, batch=256, device="cuda"):
    model.eval()
    enable_dropout(model)

    dl = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch,
        shuffle=False,
    )

    means, sqmeans = [], []
    for _ in range(n_mc):
        preds = []
        for (xb,) in dl:
            yb = model(xb.to(device))
            preds.append(yb.detach().cpu().numpy())
        yhat = np.concatenate(preds, axis=0)
        means.append(yhat)
        sqmeans.append(yhat ** 2)

    mean = np.mean(means, axis=0)
    var = np.maximum(1e-12, np.mean(sqmeans, axis=0) - mean ** 2)
    std = np.sqrt(var)
    return mean, std


def run_mcdo_inference_all_frames(
    exp_dir,
    out_dir,
    model_path,
    tau_path,
    feature_mask_path,
    scaler_path,
    indices,
    use_harmonics=1,
    n_mc=200,
    batch=256,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    feature_mask = np.load(feature_mask_path).astype(bool)
    scaler = joblib.load(scaler_path)
    model, y_mu, y_sd, sigma_scale, D_model = load_checkpoint(model_path, device=device)
    tau = float(np.load(tau_path).reshape(-1)[0])

    print(f"[INIT] device={device}")
    print(f"[INIT] Feature mask {feature_mask.sum()} / {feature_mask.size}, tau={tau:.4f}")

    all_frame_idx = []
    all_x = []
    all_y = []
    all_w = []
    all_h = []
    all_mu = []
    all_sigma = []

    for i in indices:
        exp_path = os.path.join(exp_dir, f"EFD_Seg_{int(i):04d}.npy")
        if not os.path.exists(exp_path):
            print(f"[SKIP] {exp_path} not found.")
            continue

        outputs = np.load(exp_path, allow_pickle=True)

        frame_idx_all, x_all, y_all, w_all, h_all = extract_geom(outputs, frame_idx_val=i)

        X_raw = stack_features_with_order(outputs, use_harmonics=use_harmonics)
        row_mask = np.all(np.isfinite(X_raw), axis=1)
        X_raw = X_raw[row_mask]

        frame_idx_rm = frame_idx_all[row_mask]
        x_rm = x_all[row_mask]
        y_rm = y_all[row_mask]
        w_rm = w_all[row_mask]
        h_rm = h_all[row_mask]

        if X_raw.size == 0:
            print(f"[WARN] no valid features after NaN removal for frame {int(i):04d}")
            continue

        if feature_mask.shape[0] != X_raw.shape[1]:
            print(f"[ERR] feature dim mismatch for {exp_path} ({X_raw.shape[1]} vs mask {feature_mask.shape[0]})")
            continue

        X_sel = X_raw[:, feature_mask]
        Xs = scaler.transform(X_sel).astype(np.float32)

        if Xs.shape[1] != D_model:
            print(f"[ERR] model dim mismatch for {exp_path} ({Xs.shape[1]} vs {D_model})")
            continue

        mu_std, sig_std = predict_mc(
            model,
            Xs,
            n_mc=n_mc,
            batch=batch,
            device=device,
        )

        mu = mu_std * y_sd + y_mu
        sigma = sig_std * y_sd * sigma_scale

        keep = sigma <= tau
        if not np.any(keep):
            print(f"[INFO] no samples kept (sigma <= tau) for frame {int(i):04d}")
            continue

        frame_idx_keep = frame_idx_rm[keep]
        x_keep = x_rm[keep]
        y_keep = y_rm[keep]
        w_keep = w_rm[keep]
        h_keep = h_rm[keep]
        mu_keep = mu[keep]
        sigma_keep = sigma[keep]

        all_frame_idx.append(frame_idx_keep)
        all_x.append(x_keep)
        all_y.append(y_keep)
        all_w.append(w_keep)
        all_h.append(h_keep)
        all_mu.append(mu_keep)
        all_sigma.append(sigma_keep)

        print(f"[{int(i):04d}] kept={np.mean(keep):.1%}, N_keep={len(sigma_keep)}")

    if len(all_mu) == 0:
        print("[END] No data collected. Check filters / inputs.")
        return None

    all_frame_idx = np.concatenate(all_frame_idx, axis=0)
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_w = np.concatenate(all_w, axis=0)
    all_h = np.concatenate(all_h, axis=0)
    all_mu = np.concatenate(all_mu, axis=0)
    all_sigma = np.concatenate(all_sigma, axis=0)

    out = np.column_stack(
        [
            all_frame_idx,
            all_x,
            all_y,
            all_w,
            all_h,
            all_mu,
            all_sigma,
        ]
    )

    csv_path = os.path.join(out_dir, "all_exp_pred_mcdo.csv")
    np.savetxt(
        csv_path,
        out,
        delimiter=",",
        header="frame_idx,x_centers,y_centers,bbox_widths,bbox_heights,z_predicts,z_uncertainties",
        comments="",
        fmt=["%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.6f", "%.6f"],
    )

    print(f"[DONE] saved all frames → {csv_path} (N_rows={out.shape[0]})")

    return {
        "csv_path": csv_path,
        "n_rows": int(out.shape[0]),
        "tau": tau,
    }