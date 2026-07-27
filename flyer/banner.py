"""Generate the banner figure for the recruiting flyer.

Fits three one-hidden-layer ReLU networks of increasing width to the same noisy
1-D target, so the flyer illustrates universal approximation with real fits
rather than a cartoon: three units cannot represent the curve, ten can, fifty
reach the noise floor.

Full-batch Adam reliably stalls in a V-shaped local minimum at these widths, so
each width is trained with mini-batches, a cosine schedule, and a few restarts,
keeping the best run. Results are cached in banner_fits.npz because the fits
take about a minute; delete that file (or pass --force) to retrain.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NDSU_GREEN = "#005643"
NDSU_GOLD = "#FFC72C"
GRAY = "#9AA5A2"
TARGET_GRAY = "#B9C2BF"

HERE = Path(__file__).parent
OUT = HERE / "banner.pdf"
CACHE = HERE / "banner_fits.npz"

WIDTHS = (3, 10, 50)
NOISE = 0.06
EPOCHS = 15_000
LR = 0.02
BATCH = 32
RESTARTS = 3


def target(x):
    return np.sin(2.4 * np.pi * x) * np.exp(-1.1 * x) + 0.35 * x


def make_data(n=160, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    return x, target(x) + rng.normal(0.0, NOISE, size=n)


def train_once(width, x, y, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    gen = torch.Generator().manual_seed(seed)

    for _ in range(EPOCHS):
        idx = torch.randperm(len(x), generator=gen)[:BATCH]
        opt.zero_grad()
        nn.functional.mse_loss(model(xt[idx]), yt[idx]).backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        mse = nn.functional.mse_loss(model(xt), yt).item()
        grid = torch.linspace(0, 1, 400).unsqueeze(1)
        pred = model(grid).squeeze(1).numpy()
    return mse, grid.squeeze(1).numpy(), pred


def best_fit(width, x, y):
    runs = [train_once(width, x, y, seed) for seed in range(RESTARTS)]
    mse, gx, gy = min(runs, key=lambda r: r[0])
    print(f"  width {width:>3}: mse {mse:.4f}  (restarts {[round(r[0], 4) for r in runs]})")
    return gx, gy, mse


def compute_fits(x, y):
    print("training banner fits")
    fits = {}
    for width in WIDTHS:
        gx, gy, mse = best_fit(width, x, y)
        fits[f"gx{width}"] = gx
        fits[f"gy{width}"] = gy
        fits[f"mse{width}"] = np.array(mse)
    np.savez(CACHE, **fits)
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="retrain, ignoring cache")
    args = ap.parse_args()

    x, y = make_data()
    if CACHE.exists() and not args.force:
        fits = dict(np.load(CACHE))
        print(f"using cached fits from {CACHE.name}")
    else:
        fits = compute_fits(x, y)

    grid = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 1.6), sharey=True)

    for ax, width in zip(axes, WIDTHS):
        ax.plot(grid, target(grid), color=TARGET_GRAY, lw=1.5, ls=(0, (3, 2)), zorder=2)
        ax.scatter(x, y, s=5, color=GRAY, alpha=0.7, linewidths=0, zorder=3)
        ax.plot(fits[f"gx{width}"], fits[f"gy{width}"], color=NDSU_GREEN, lw=2.2, zorder=4)
        ax.set_title(
            f"{width} hidden units", fontsize=9, color=NDSU_GREEN, pad=4, fontweight="bold"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(NDSU_GOLD)
        ax.spines["bottom"].set_linewidth(1.6)

    axes[0].set_ylim(-0.62, 1.08)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.85, bottom=0.06, wspace=0.07)
    fig.savefig(OUT, format="pdf", transparent=True)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
