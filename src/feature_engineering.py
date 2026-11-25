"""Chronological splits and fixed-length forecasting sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CALENDAR_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_day_sin",
    "year_day_cos",
)


@dataclass
class SequenceSplit:
    x: np.ndarray
    y: np.ndarray
    target_times: pd.DatetimeIndex


def chronological_boundaries(index: pd.DatetimeIndex, train_fraction: float = 0.70, validation_fraction: float = 0.15):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must sum to less than one.")
    first = max(1, int(len(index) * train_fraction))
    second = max(first + 1, int(len(index) * (train_fraction + validation_fraction)))
    if second >= len(index):
        raise ValueError("Not enough timestamps for chronological train/validation/test splits.")
    return index[first], index[second]


def fit_scale_train_only(frame: pd.DataFrame, train_end: pd.Timestamp):
    """Fit one scale per sensor using observations strictly before train_end."""
    train = frame.loc[frame.index < train_end]
    if train.empty:
        raise ValueError("No observations available to fit the training scaler.")
    scaler = StandardScaler()
    scaler.fit(train.to_numpy(dtype=np.float32))
    values = scaler.transform(frame.to_numpy(dtype=np.float32)).astype(np.float32)
    return values, scaler


def build_model_features(scaled_values: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """Append cyclical hour, weekday, and annual time features to each sensor."""
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    weekday = index.dayofweek.to_numpy() + hour / 24.0
    year_day = index.dayofyear.to_numpy() - 1 + hour / (24.0 * 365.25)
    phases = (
        (hour, 24.0),
        (weekday, 7.0),
        (year_day, 365.25),
    )
    calendar = np.stack(
        [trig(2 * np.pi * values / period) for values, period in phases for trig in (np.sin, np.cos)],
        axis=-1,
    ).astype(np.float32)
    repeated = np.broadcast_to(calendar[:, None, :], (len(index), scaled_values.shape[1], calendar.shape[1]))
    return np.concatenate((scaled_values[..., None], repeated), axis=-1).astype(np.float32)


def make_sequences(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    lookback: int,
    horizon_steps: int,
    target_values: np.ndarray | None = None,
) -> SequenceSplit:
    if lookback < 1 or horizon_steps < 1:
        raise ValueError("lookback and horizon_steps must be positive.")
    if len(values) != len(index):
        raise ValueError("Values and timestamp index must have the same number of rows.")
    if values.ndim == 2:
        features = values[..., np.newaxis]
        targets = values
    elif values.ndim == 3:
        features = values
        targets = values[..., 0] if target_values is None else target_values
    else:
        raise ValueError("Values must have shape (timestamps, sensors[, features]).")
    if targets.shape[:2] != values.shape[:2]:
        raise ValueError("target_values must match the timestamp and sensor dimensions.")
    stop = len(values) - horizon_steps
    origins = np.arange(lookback - 1, stop)
    if len(origins) == 0:
        raise ValueError("Dataset is too short for the requested lookback and horizon.")
    xs = np.stack([features[i - lookback + 1 : i + 1] for i in origins])
    target_indices = origins + horizon_steps
    ys = targets[target_indices]
    return SequenceSplit(xs.astype(np.float32), ys.astype(np.float32), index[target_indices])


def partition_sequences(sequences: SequenceSplit, train_end: pd.Timestamp, validation_end: pd.Timestamp):
    times = sequences.target_times
    masks = (
        times < train_end,
        (times >= train_end) & (times < validation_end),
        times >= validation_end,
    )
    return tuple(
        SequenceSplit(sequences.x[mask], sequences.y[mask], times[mask])
        for mask in masks
    )
