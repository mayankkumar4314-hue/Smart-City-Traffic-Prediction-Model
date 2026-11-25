"""Build sensor graphs from coordinates or training-only speed correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


def adjacency_from_coordinates(sensor_ids, coordinates: pd.DataFrame, distance_km: float = 5.0) -> np.ndarray:
    required = {"sensor_id", "latitude", "longitude"}
    if not required.issubset(coordinates.columns):
        raise ValueError(f"Coordinate table must contain columns {sorted(required)}.")
    indexed = coordinates.drop_duplicates("sensor_id").set_index("sensor_id")
    indexed.index = indexed.index.map(str)
    ordered_sensor_ids = [str(sensor) for sensor in sensor_ids]
    missing = [sensor for sensor in ordered_sensor_ids if sensor not in indexed.index]
    if missing:
        raise ValueError(f"Coordinates are missing {len(missing)} sensor IDs, e.g. {missing[:5]}.")
    ordered = indexed.loc[ordered_sensor_ids, ["latitude", "longitude"]].astype(float)
    radians = np.radians(ordered.to_numpy())
    distances = haversine_distances(radians) * 6371.0088
    sigma = max(distance_km, 1e-6)
    adjacency = np.exp(-np.square(distances / sigma))
    adjacency[distances > distance_km] = 0.0
    np.fill_diagonal(adjacency, 1.0)
    return _symmetric_normalize(adjacency)


def adjacency_from_training_speeds(training_frame: pd.DataFrame, threshold: float = 0.3) -> np.ndarray:
    """Estimate a nonnegative correlation graph using only the training period."""
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0, 1).")
    corr = training_frame.corr().fillna(0).to_numpy(dtype=np.float32)
    adjacency = np.abs(corr)
    adjacency[adjacency < threshold] = 0.0
    np.fill_diagonal(adjacency, 1.0)
    return _symmetric_normalize(adjacency)


def _symmetric_normalize(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1e-8), -0.5)
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return normalized.astype(np.float32)
