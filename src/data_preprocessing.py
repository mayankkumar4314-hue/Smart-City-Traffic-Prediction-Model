"""Load and clean traffic sensor observations without inventing measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
SENSOR_CANDIDATES = ("sensor_id", "station_id", "location_id", "node_id")
METRIC_NAMES = ("speed", "flow", "occupancy", "density")


def _find_column(columns: pd.Index, requested: str | None, candidates: tuple[str, ...]) -> str | None:
    if requested:
        if requested not in columns:
            raise ValueError(f"Column {requested!r} is not present. Available columns: {list(columns)}")
        return requested
    by_lower = {str(column).lower(): str(column) for column in columns}
    return next((by_lower[name] for name in candidates if name in by_lower), None)


def load_traffic_data(
    path: str | Path,
    timestamp_col: str | None = None,
    sensor_col: str | None = None,
    target: str = "speed",
    hdf_key: str = "df",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    """Return a timestamp-indexed, sensor-column target frame and quality summary.

    CSV input may be long form (timestamp/sensor/target) or wide form (timestamp
    and one column per sensor). HDF5 input follows the common benchmark layout:
    timestamps in rows and sensor IDs in columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Traffic dataset not found: {path}")

    if path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        frame = pd.read_hdf(path, key=hdf_key)
        if isinstance(frame.index, pd.MultiIndex):
            raise ValueError("Expected a single timestamp index in the HDF5 table.")
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[frame.index.notna()]
        frame.index.name = "timestamp"
        coordinates = None
        raw_count = int(frame.size)
        frame = frame.apply(pd.to_numeric, errors="coerce")
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        time_col = _find_column(raw.columns, timestamp_col, TIMESTAMP_CANDIDATES)
        if time_col is None:
            raise ValueError(f"Could not identify timestamp column in {path.name}.")
        sensor = _find_column(raw.columns, sensor_col, SENSOR_CANDIDATES)
        parsed_time = pd.to_datetime(raw[time_col], errors="coerce", utc=True).dt.tz_convert(None)
        raw = raw.loc[parsed_time.notna()].copy()
        raw[time_col] = parsed_time.loc[parsed_time.notna()]
        coordinates = None
        lower_to_original = {str(column).lower(): str(column) for column in raw.columns}
        lat_col = lower_to_original.get("latitude", lower_to_original.get("lat"))
        lon_col = lower_to_original.get("longitude", lower_to_original.get("lon", lower_to_original.get("lng")))
        if sensor and lat_col and lon_col:
            coordinates = (
                raw[[sensor, lat_col, lon_col]]
                .dropna()
                .drop_duplicates(sensor, keep="last")
                .rename(columns={sensor: "sensor_id", lat_col: "latitude", lon_col: "longitude"})
            )
            coordinates["sensor_id"] = coordinates["sensor_id"].astype(str)
        raw_count = int(len(raw))
        if sensor:
            if target not in raw.columns:
                raise ValueError(f"Target column {target!r} is not present in the long-form CSV.")
            selected = raw[[time_col, sensor, target]].copy()
            selected[target] = pd.to_numeric(selected[target], errors="coerce")
            selected = selected.drop_duplicates([time_col, sensor], keep="last")
            frame = selected.pivot(index=time_col, columns=sensor, values=target)
        else:
            reserved = {time_col}
            reserved.update(column for column in (lat_col, lon_col) if column)
            if target in raw.columns:
                raise ValueError(
                    f"Found {target!r} without a sensor ID. Use long form with a sensor column "
                    "or a wide CSV with one sensor per column."
                )
            value_columns = [column for column in raw.columns if column not in reserved]
            if not value_columns:
                raise ValueError("No sensor value columns found in wide-form CSV.")
            frame = raw.set_index(time_col)[value_columns].apply(pd.to_numeric, errors="coerce")
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[frame.index.notna()]
        frame.index.name = "timestamp"
    else:
        raise ValueError("Supported dataset formats are .csv, .h5, .hdf, and .hdf5.")

    frame = frame.sort_index()
    frame.columns = frame.columns.map(str)
    duplicates = int(frame.index.duplicated(keep="last").sum())
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    before_invalid = int(frame.isna().sum().sum())
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if target.lower() == "speed":
        frame = frame.mask((frame < 0) | (frame > 160))

    summary = {
        "source": str(path),
        "rows": raw_count,
        "timestamps": int(len(frame)),
        "sensors": int(frame.shape[1]),
        "duplicate_timestamps_removed": duplicates,
        "missing_values_before_resampling": before_invalid,
        "target": target,
        "start": frame.index.min().isoformat() if len(frame) else None,
        "end": frame.index.max().isoformat() if len(frame) else None,
    }
    return frame, summary, coordinates


def regularize_series(frame: pd.DataFrame, interval_minutes: int = 5, fill_limit: int = 2) -> pd.DataFrame:
    """Resample to a fixed interval and fill only short gaps from past values."""
    if interval_minutes <= 0 or fill_limit < 0:
        raise ValueError("interval_minutes must be positive and fill_limit cannot be negative.")
    if frame.empty:
        raise ValueError("No usable traffic observations remain after loading.")
    regular = frame.sort_index().resample(f"{interval_minutes}min").mean()
    if fill_limit:
        regular = regular.ffill(limit=fill_limit)
    regular = regular.dropna(axis=0, how="any")
    if regular.empty:
        raise ValueError("No complete rows remain after resampling and bounded gap filling.")
    return regular
