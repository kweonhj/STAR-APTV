import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_calib(path):
    data = np.load(path)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return X[m], y[m]


def standardize_y(y: np.ndarray):
    mu = float(np.mean(y))
    sd = float(np.std(y) + 1e-8)
    return ((y - mu) / sd).astype(np.float32), mu, sd


def unstandardize(v: np.ndarray, mu: float, sd: float):
    return v * sd + mu


def eval_report(mu: np.ndarray, sig: np.ndarray, y_true: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(y_true, mu)))
    r2 = float(r2_score(y_true, mu))
    cov2 = float(np.mean(np.abs(y_true - mu) <= 2.0 * np.maximum(sig, 1e-12))) if len(y_true) else float("nan")
    avgS = float(np.nanmean(sig)) if len(sig) else float("nan")
    return {
        "rmse": rmse,
        "r2": r2,
        "cov_2sigma": cov2,
        "avg_sigma": avgS,
    }


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


def enable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_mc(model, X, n_mc=200, batch=128, device="cuda"):
    model.eval()
    enable_dropout(model)

    dl = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch, shuffle=False)
    means, sqmeans = [], []

    for _ in range(n_mc):
        preds = []
        for (xb,) in dl:
            xb = xb.to(device)
            yb = model(xb)
            preds.append(yb.detach().cpu().numpy())
        yhat = np.concatenate(preds, axis=0)
        means.append(yhat)
        sqmeans.append(yhat ** 2)

    mean = np.mean(means, axis=0)
    var = np.maximum(1e-12, np.mean(sqmeans, axis=0) - mean ** 2)
    std = np.sqrt(var)
    return mean, std


def train_epoch(model, dl, opt, loss_fn, device="cuda"):
    model.train()
    tot = 0.0
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
    return tot / len(dl.dataset)


@torch.no_grad()
def evaluate(model, dl, loss_fn, device="cuda"):
    model.eval()
    tot = 0.0
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        tot += loss.item() * xb.size(0)
    return tot / len(dl.dataset)


def run_mc_dropout(
    calib_path,
    model_path,
    tau_path,
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
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)
    print(f"[DEVICE] {device}")

    X, y = load_calib(calib_path)
    y_std, y_mu, y_sd = standardize_y(y)

    D = X.shape[1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_std, test_size=test_size, random_state=random_state
    )

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch,
        shuffle=True,
        drop_last=False,
    )
    te_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=batch,
        shuffle=False,
        drop_last=False,
    )

    model = MLP_MCDropout(D, list(hidden), p=dropout_p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss(delta=huber_delta) if use_huber else nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=100,
        T_mult=2,
        eta_min=lr * 0.05,
    )

    best_val = float("inf")
    best_state = None
    patience_left = patience

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, tr_dl, opt, loss_fn, device=device)
        val_loss = evaluate(model, te_dl, loss_fn, device=device)
        scheduler.step(ep)

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if ep % 50 == 0 or ep == 1:
            print(f"[{ep:04d}] train={tr_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if patience_left <= 0:
            print(f"[ES] Early stop at {ep}, best val={best_val:.4f}")
            break

    print(f"[FIT] done in {time.time() - t0:.1f}s; best val loss={best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    mu_te_std, sig_te_std = predict_mc(
        model,
        X_te,
        n_mc=n_mc,
        batch=batch,
        device=device,
    )

    mu_te = unstandardize(mu_te_std, y_mu, y_sd)
    sig_te = sig_te_std * y_sd
    y_te_o = unstandardize(y_te, y_mu, y_sd)

    sigma_scale = 1.0
    if use_sig_cal and len(y_te_o):
        r = np.abs(y_te_o - mu_te) / np.maximum(sig_te, 1e-12)
        q = np.quantile(r, cover_target)
        sigma_scale = q / 2.0
        sig_te = sig_te * sigma_scale

    rep = eval_report(mu_te, sig_te, y_te_o)
    tau = float(np.quantile(sig_te, 0.99)) if len(sig_te) else float("nan")

    print(f"[SPLIT] Ntr={len(y_tr)}, Nte={len(y_te)}")
    print(f"[SPLIT] RMSE={rep['rmse']:.4g}, R^2={rep['r2']:.3f}, 2σ={rep['cov_2sigma']:.2%}, avgσ={rep['avg_sigma']:.4g}")
    print(f"[TAU]   q=0.99 → τ={tau:.4g}  (sigma_scale={sigma_scale:.3f})")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": D,
            "hidden": list(hidden),
            "dropout_p": dropout_p,
            "y_mu": y_mu,
            "y_sd": y_sd,
            "sigma_scale": float(sigma_scale),
        },
        model_path,
    )

    os.makedirs(os.path.dirname(tau_path), exist_ok=True)
    np.save(tau_path, np.array([tau], dtype=np.float32))

    print(f"[SAVED] model → {model_path}")
    print(f"[SAVED] tau   → {tau_path}")

    return {
        "rmse": rep["rmse"],
        "r2": rep["r2"],
        "cov_2sigma": rep["cov_2sigma"],
        "avg_sigma": rep["avg_sigma"],
        "tau": tau,
        "sigma_scale": sigma_scale,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "best_val_loss": best_val,
    }