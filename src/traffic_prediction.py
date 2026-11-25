"""Load trained artifacts and produce predictions from the newest historical window."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import build_model_features


NEURAL_MODELS = {"lstm", "gru", "cnn_lstm", "gcn_lstm", "stgcn"}


def load_prediction_bundle(model_dir: str | Path):
    directory = Path(model_dir)
    required = (directory / "metadata.json", directory / "scaler.joblib", directory / "selected_models.csv", directory / "adjacency.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Trained model bundle is incomplete; missing: {', '.join(missing)}")
    with (directory / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    return {
        "directory": directory,
        "metadata": metadata,
        "scaler": joblib.load(directory / "scaler.joblib"),
        "selected": pd.read_csv(directory / "selected_models.csv"),
        "adjacency": np.load(directory / "adjacency.npy"),
    }


def predict_future(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    metadata = bundle["metadata"]
    sensors = [str(sensor) for sensor in metadata["sensor_ids"]]
    missing = [sensor for sensor in sensors if sensor not in frame.columns]
    if missing:
        raise ValueError(f"Input data does not contain model sensors: {missing[:5]}")
    ordered = frame.loc[:, sensors].sort_index().tail(metadata["lookback"])
    if len(ordered) < metadata["lookback"] or ordered.isna().any().any():
        raise ValueError("Not enough complete recent observations to construct the model input window.")
    scaler = bundle["scaler"]
    normalized = scaler.transform(ordered.to_numpy(dtype=np.float32)).astype(np.float32)
    features = build_model_features(normalized, ordered.index)
    x = features[np.newaxis, ...]
    interval = int(metadata["interval_minutes"])
    origin = ordered.index[-1]
    rows = []
    for _, selected in bundle["selected"].iterrows():
        horizon_minutes = int(selected["horizon_minutes"])
        model_name = str(selected["model"])
        path = bundle["directory"] / f"{model_name}_{horizon_minutes}m"
        if model_name in NEURAL_MODELS:
            import torch

            from src.models import build_neural_model

            model = build_neural_model(model_name, bundle["adjacency"], x.shape[-1])
            state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                normalized_prediction = model(torch.from_numpy(x)).numpy()
        else:
            estimator = joblib.load(path.with_suffix(".joblib"))
            normalized_prediction = estimator.predict(x.reshape(1, -1))
        values = scaler.inverse_transform(normalized_prediction)[0]
        if metadata["target"].lower() == "speed":
            values = np.clip(values, 0, 160)
        for sensor, value in zip(sensors, values):
            rows.append({
                "sensor_id": sensor,
                "horizon_minutes": horizon_minutes,
                "predicted_at": origin,
                "forecast_for": origin + pd.Timedelta(minutes=horizon_minutes),
                "target": metadata["target"],
                "predicted_value": float(value),
                "model": model_name,
            })
    return pd.DataFrame(rows)
