"""Forecast metrics and plots for held-out sensor-time observations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        raise ValueError("No finite actual/predicted values available for evaluation.")
    nonzero = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": mape,
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def save_prediction_plot(times, actual, predicted, output_path: str | Path, title: str):
    actual_mean = np.asarray(actual).mean(axis=1)
    predicted_mean = np.asarray(predicted).mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, actual_mean, label="Observed sensor mean", linewidth=1.2)
    ax.plot(times, predicted_mean, label="Predicted sensor mean", linewidth=1.2)
    ax.set(title=title, xlabel="Target timestamp", ylabel="Target value")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_metrics(metrics: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_path, index=False)
