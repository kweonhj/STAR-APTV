# plot_result.py
# ------------------------------------------------------------
# Plot MCDO inference results from all_exp_pred_mcdo.csv
# ------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_result_mcdo/all_exp_pred_mcdo.csv"
OUT_DIR = "/data1/jaejung/others/hyeokjin/sam2/notebooks/ForGit/Experiment/LD_exp_result_mcdo/plots"


def load_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    required_cols = [
        "frame_idx",
        "x_centers",
        "y_centers",
        "bbox_widths",
        "bbox_heights",
        "z_predicts",
        "z_uncertainties",
    ]
    for col in required_cols:
        if col not in data.dtype.names:
            raise ValueError(f"Missing column in CSV: {col}")

    return data


def compute_sizes(w, h):
    area_root = np.sqrt(np.maximum(w * h, 1e-12))
    scale = np.nanpercentile(area_root, 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    sizes = 10 + area_root / scale * 80
    return sizes


def set_equal_aspect_3d(ax, X, Y, Z):
    max_range = np.array(
        [
            np.nanmax(X) - np.nanmin(X),
            np.nanmax(Y) - np.nanmin(Y),
            np.nanmax(Z) - np.nanmin(Z),
        ]
    ).max() / 2.0

    mid_x = (np.nanmax(X) + np.nanmin(X)) * 0.5
    mid_y = (np.nanmax(Y) + np.nanmin(Y)) * 0.5
    mid_z = (np.nanmax(Z) + np.nanmin(Z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_3d_scatter(frame_idx, x, y, z, sizes, out_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x,
        y,
        z,
        c=frame_idx,
        cmap="viridis",
        s=sizes,
        alpha=0.7,
        depthshade=True,
    )

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Frame Index", fontsize=12)

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("Predicted z")
    ax.set_title("3D MCDO Predictions", fontsize=13)

    set_equal_aspect_3d(ax, x, y, z)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_view(frame_idx, x, y, z, sizes, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        x,
        y,
        c=z,
        s=sizes,
        cmap="plasma",
        alpha=0.75,
    )

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Predicted z", fontsize=11)

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Top View: x-y colored by predicted z")

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_side_view(frame_idx, x, y, z, sizes, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(
        x,
        z,
        c=frame_idx,
        s=sizes,
        cmap="viridis",
        alpha=0.75,
    )

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Frame Index", fontsize=11)

    ax.set_xlabel("x (px)")
    ax.set_ylabel("Predicted z")
    ax.set_title("Side View: x-z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uncertainty_histogram(unc, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(unc, bins=40, edgecolor="black")
    ax.set_xlabel("Predicted uncertainty")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of MCDO uncertainty")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_framewise_summary(frame_idx, z, unc, out_path):
    uniq = np.unique(frame_idx).astype(int)

    z_mean = []
    z_std = []
    u_mean = []
    counts = []

    for i in uniq:
        m = frame_idx == i
        z_mean.append(np.nanmean(z[m]))
        z_std.append(np.nanstd(z[m]))
        u_mean.append(np.nanmean(unc[m]))
        counts.append(np.sum(m))

    z_mean = np.array(z_mean)
    z_std = np.array(z_std)
    u_mean = np.array(u_mean)
    counts = np.array(counts)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(uniq, z_mean)
    axes[0].fill_between(uniq, z_mean - z_std, z_mean + z_std, alpha=0.25)
    axes[0].set_ylabel("z mean ± std")
    axes[0].set_title("Frame-wise prediction summary")

    axes[1].plot(uniq, u_mean)
    axes[1].set_ylabel("mean uncertainty")

    axes[2].plot(uniq, counts)
    axes[2].set_ylabel("kept count")
    axes[2].set_xlabel("frame index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_csv(CSV_PATH)

    frame_idx = data["frame_idx"]
    x = data["x_centers"]
    y = data["y_centers"]
    w = data["bbox_widths"]
    h = data["bbox_heights"]
    z = data["z_predicts"]
    unc = data["z_uncertainties"]

    sizes = compute_sizes(w, h)

    print(f"[INFO] Loaded rows: {len(z)}")
    print(f"[INFO] z range: {np.nanmin(z):.4f} ~ {np.nanmax(z):.4f}")
    print(f"[INFO] uncertainty range: {np.nanmin(unc):.4f} ~ {np.nanmax(unc):.4f}")

    plot_3d_scatter(
        frame_idx, x, y, z, sizes,
        os.path.join(OUT_DIR, "plot_3d_scatter.png"),
    )

    plot_top_view(
        frame_idx, x, y, z, sizes,
        os.path.join(OUT_DIR, "plot_top_view_xy_zcolor.png"),
    )

    plot_side_view(
        frame_idx, x, y, z, sizes,
        os.path.join(OUT_DIR, "plot_side_view_xz.png"),
    )

    plot_uncertainty_histogram(
        unc,
        os.path.join(OUT_DIR, "plot_uncertainty_hist.png"),
    )

    plot_framewise_summary(
        frame_idx, z, unc,
        os.path.join(OUT_DIR, "plot_framewise_summary.png"),
    )

    print(f"[DONE] plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()