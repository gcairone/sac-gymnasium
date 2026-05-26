from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("TkAgg")  # o "Qt5Agg"
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAC training logs.")
    parser.add_argument("--log_dir", required=True, help="Path to a run directory")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing")
    return parser.parse_args()


def load_csv(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        raise ValueError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    header = rows[0]
    if not header:
        raise ValueError(f"CSV missing header: {path}")

    data: Dict[str, List[float]] = {name: [] for name in header}
    for row in rows[1:]:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"CSV row has wrong length in {path}")
        for idx, value in enumerate(row):
            data[header[idx]].append(float(value))
    return data


def smooth(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    if not values:
        return []

    left = window // 2
    right = window - 1 - left
    padded = [values[0]] * left + list(values) + [values[-1]] * right

    out: List[float] = []
    for idx in range(len(values)):
        start = idx
        end = idx + window
        segment = padded[start:end]
        out.append(sum(segment) / float(window))
    return out


def plot_train_metrics(
    data: Dict[str, List[float]],
    smooth_w: int,
    save: bool,
    out_dir: Path,
) -> None:
    if "step" not in data:
        raise ValueError("train.csv missing 'step' column")

    steps = data["step"]
    series = {
        "actor_loss": data.get("actor_loss", []),
        "critic_loss": data.get("critic_loss", []),
        "alpha_loss": data.get("alpha_loss", []),
        "alpha": data.get("alpha", []),
        "entropy": data.get("entropy", []),
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, values) in enumerate(series.items()):
        if not values:
            axes[idx].set_title(f"{name} (missing)")
            axes[idx].axis("off")
            continue
        y = smooth(values, smooth_w)
        axes[idx].plot(steps, y)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("step")

    axes[-1].axis("off")
    fig.tight_layout()

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "train_metrics.png", dpi=150)
    else:
        plt.show()


def plot_eval_metrics(
    data: Dict[str, List[float]],
    smooth_w: int,
    save: bool,
    out_dir: Path,
) -> None:
    if "step" not in data:
        raise ValueError("eval.csv missing 'step' column")
    if "mean_return" not in data:
        raise ValueError("eval.csv missing 'mean_return' column")

    steps = data["step"]
    mean_return = smooth(data["mean_return"], smooth_w)
    std_return = data.get("std_return", [0.0] * len(mean_return))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(steps, mean_return, label="mean_return")
    if len(std_return) == len(mean_return):
        lower = [m - s for m, s in zip(mean_return, std_return)]
        upper = [m + s for m, s in zip(mean_return, std_return)]
        ax.fill_between(steps, lower, upper, alpha=0.2)

    ax.set_title("Evaluation")
    ax.set_xlabel("step")
    ax.set_ylabel("return")
    ax.legend()
    fig.tight_layout()

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "eval_metrics.png", dpi=150)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise ValueError(f"log_dir not found: {log_dir}")

    train_path = log_dir / "train.csv"
    eval_path = log_dir / "eval.csv"

    if train_path.exists():
        train_data = load_csv(train_path)
        plot_train_metrics(train_data, args.smooth, args.save, log_dir)

    if eval_path.exists():
        eval_data = load_csv(eval_path)
        plot_eval_metrics(eval_data, args.smooth, args.save, log_dir)

    if not train_path.exists() and not eval_path.exists():
        raise ValueError("No train.csv or eval.csv found in log_dir")


if __name__ == "__main__":
    main()
