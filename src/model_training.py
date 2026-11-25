"""Train comparable forecasting models with chronological validation and testing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_preprocessing import load_traffic_data, regularize_series
from src.feature_engineering import (
    CALENDAR_FEATURE_NAMES,
    build_model_features,
    chronological_boundaries,
    fit_scale_train_only,
    make_sequences,
    partition_sequences,
)
from src.graph_construction import adjacency_from_coordinates, adjacency_from_training_speeds
from src.model_evaluation import calculate_metrics, save_metrics, save_prediction_plot
from src.models import build_neural_model

DEFAULT_MODELS = ("linear", "random_forest", "lstm", "gcn_lstm", "stgcn")
NEURAL_MODELS = {"lstm", "gru", "cnn_lstm", "gcn_lstm", "stgcn"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _baseline(name: str, seed: int):
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "svr":
        return MultiOutputRegressor(SVR(C=10.0, epsilon=0.1))
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost was requested but is not installed. Install xgboost or omit it from --models.") from exc
        return MultiOutputRegressor(XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, objective="reg:squarederror", random_state=seed, n_jobs=-1))
    raise ValueError(f"Unknown baseline model: {name}")


def _flat(x):
    return x.reshape(len(x), -1)


def _train_neural(name, train, validation, adjacency, epochs, batch_size, seed, device):
    set_seed(seed)
    model = build_neural_model(name, adjacency, train.x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    train_data = TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    x_val = torch.from_numpy(validation.x).to(device)
    y_val = torch.from_numpy(validation.y).to(device)
    best_loss, best_state, patience = float("inf"), None, 6
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 6
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError(f"Neural model {name} did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_training(args):
    set_seed(args.seed)
    raw, quality, coordinates = load_traffic_data(args.data, args.timestamp_col, args.sensor_col, args.target, args.hdf_key)
    frame = regularize_series(raw, args.interval_minutes)
    train_end, validation_end = chronological_boundaries(frame.index)
    scaled, scaler = fit_scale_train_only(frame, train_end)
    model_features = build_model_features(scaled, frame.index)
    sensor_ids = list(frame.columns)
    train_frame = frame.loc[frame.index < train_end]
    if args.coordinates_csv:
        coordinates = pd.read_csv(args.coordinates_csv)
    if coordinates is not None:
        if "sensor_id" not in coordinates.columns:
            raise ValueError("Coordinate CSV must include a sensor_id column.")
        coordinates = coordinates.copy()
        coordinates["sensor_id"] = coordinates["sensor_id"].astype(str)
        adjacency = adjacency_from_coordinates(sensor_ids, coordinates)
        graph_source = "sensor coordinates, distance-kernel adjacency"
    else:
        adjacency = adjacency_from_training_speeds(train_frame)
        graph_source = "training-only absolute speed correlation"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output / "scaler.joblib")
    with (output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({
            "target": args.target,
            "interval_minutes": args.interval_minutes,
            "lookback": args.lookback,
            "calendar_features": list(CALENDAR_FEATURE_NAMES),
            "sensor_ids": sensor_ids,
            "train_end_exclusive": train_end.isoformat(),
            "validation_end_exclusive": validation_end.isoformat(),
            "graph_source": graph_source,
            "quality": quality,
        }, file, indent=2)
    np.save(output / "adjacency.npy", adjacency)

    all_metrics = []
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    for horizon_minutes in args.horizons:
        steps = max(1, int(round(horizon_minutes / args.interval_minutes)))
        sequences = make_sequences(model_features, frame.index, args.lookback, steps, target_values=scaled)
        train, validation, test = partition_sequences(sequences, train_end, validation_end)
        if min(len(train.x), len(validation.x), len(test.x)) == 0:
            raise ValueError(f"Insufficient complete sequences for {horizon_minutes}-minute horizon.")
        horizon_best = (float("inf"), None)
        for name in model_names:
            artifact = output / f"{name}_{horizon_minutes}m"
            if name in NEURAL_MODELS:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = _train_neural(name, train, validation, adjacency, args.epochs, args.batch_size, args.seed, device)
                model.to("cpu")
                torch.save(model.state_dict(), artifact.with_suffix(".pt"))
                model.to(device)

                def predict(split):
                    model.eval()
                    result = []
                    loader = DataLoader(TensorDataset(torch.from_numpy(split.x)), batch_size=args.batch_size)
                    with torch.no_grad():
                        for (batch,) in loader:
                            result.append(model(batch.to(device)).cpu().numpy())
                    return np.concatenate(result)
            else:
                estimator = _baseline(name, args.seed)
                estimator.fit(_flat(train.x), train.y)
                joblib.dump(estimator, artifact.with_suffix(".joblib"))

                def predict(split, fitted=estimator):
                    return fitted.predict(_flat(split.x))

            val_prediction = predict(validation)
            val_metrics = calculate_metrics(validation.y, val_prediction)
            test_prediction = predict(test)
            test_actual = scaler.inverse_transform(test.y)
            test_prediction_original = scaler.inverse_transform(test_prediction)
            if args.target.lower() == "speed":
                test_prediction_original = np.clip(test_prediction_original, 0, 160)
            row = {
                "model": name,
                "horizon_minutes": horizon_minutes,
                "validation_mae_scaled": val_metrics["mae"],
                **calculate_metrics(test_actual, test_prediction_original),
            }
            all_metrics.append(row)
            if val_metrics["mae"] < horizon_best[0]:
                horizon_best = (val_metrics["mae"], name)
            save_prediction_plot(
                test.target_times, test_actual, test_prediction_original,
                output / "plots" / f"{name}_{horizon_minutes}m.png",
                f"{name}: {horizon_minutes}-minute forecast (held-out test)",
            )
        if horizon_best[1] is None:
            raise RuntimeError(f"No model was successfully trained for {horizon_minutes} minutes.")
        print(f"Selected {horizon_best[1]} for {horizon_minutes}-minute forecasts by validation MAE.")

    save_metrics(all_metrics, output / "metrics.csv")
    metrics_frame = pd.DataFrame(all_metrics)
    selected = metrics_frame.loc[metrics_frame.groupby("horizon_minutes")["validation_mae_scaled"].idxmin()]
    selected.to_csv(output / "selected_models.csv", index=False)
    print(f"\nData quality: {json.dumps(quality, indent=2)}")
    print(f"\nHeld-out test metrics:\n{metrics_frame.to_string(index=False)}")
    print(f"\nArtifacts and reports saved to: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to CSV or HDF5 traffic data.")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--sensor-col", default=None)
    parser.add_argument("--target", default="speed")
    parser.add_argument("--hdf-key", default="df")
    parser.add_argument("--coordinates-csv", default=None, help="Optional CSV with sensor_id, latitude, longitude.")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
