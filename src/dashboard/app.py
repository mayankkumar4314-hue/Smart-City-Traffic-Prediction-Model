"""Historical traffic analysis and trained-model forecast dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_traffic_data, regularize_series
from src.traffic_prediction import load_prediction_bundle, predict_future


st.set_page_config(page_title="Smart City Traffic Forecast", page_icon="🚦", layout="wide")


@st.cache_data(show_spinner=False)
def load_local_dataset(path: str, target: str, interval: int, source_version: int):
    _ = source_version
    wide, quality, coords = load_traffic_data(path, target=target)
    return regularize_series(wide, interval), quality, coords


st.title("Smart City Traffic Forecast")
st.caption("Historical sensor analysis and model-based forecasts. No live traffic feed is connected.")

with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Upload traffic CSV", type=["csv"])
    default_data = PROJECT_ROOT / "datasets" / "traffic.csv"
    data_path = st.text_input("Or local CSV/HDF5 path", str(default_data))
    interval_minutes = st.number_input("Sampling interval (minutes)", min_value=1, value=5)
    target_name = st.text_input("Target metric", "speed")
    st.caption("CSV: long form (timestamp, sensor_id, metric) or wide form (timestamp + sensor columns).")

try:
    if upload is not None:
        uploaded = pd.read_csv(upload)
        time_candidates = {str(c).lower(): c for c in uploaded.columns}
        time_col = next((time_candidates[name] for name in ("timestamp", "datetime", "date", "time") if name in time_candidates), None)
        if time_col is None:
            st.error("CSV must include a timestamp, datetime, date, or time column.")
            st.stop()
        sensor_col = next((time_candidates[name] for name in ("sensor_id", "station_id", "location_id", "node_id") if name in time_candidates), None)
        if sensor_col:
            uploaded[time_col] = pd.to_datetime(uploaded[time_col], errors="coerce", utc=True).dt.tz_convert(None)
            uploaded = uploaded.loc[uploaded[time_col].notna()].copy()
            uploaded[target_name] = pd.to_numeric(uploaded[target_name], errors="coerce")
            if target_name.lower() == "speed":
                uploaded[target_name] = uploaded[target_name].mask((uploaded[target_name] < 0) | (uploaded[target_name] > 160))
            uploaded = uploaded.drop_duplicates([time_col, sensor_col], keep="last")
            wide = uploaded.pivot_table(index=time_col, columns=sensor_col, values=target_name, aggfunc="last")
            coordinates = uploaded[[sensor_col, *[time_candidates[name] for name in ("latitude", "longitude") if name in time_candidates]]].drop_duplicates(sensor_col) if {"latitude", "longitude"}.issubset(time_candidates) else None
            if coordinates is not None:
                coordinates = coordinates.rename(columns={sensor_col: "sensor_id", time_candidates["latitude"]: "latitude", time_candidates["longitude"]: "longitude"})
                coordinates["sensor_id"] = coordinates["sensor_id"].astype(str)
        else:
            uploaded[time_col] = pd.to_datetime(uploaded[time_col], errors="coerce", utc=True).dt.tz_convert(None)
            uploaded = uploaded.loc[uploaded[time_col].notna()].drop_duplicates(time_col, keep="last")
            wide = uploaded.set_index(time_col)
            coordinates = None
        wide.index = pd.to_datetime(wide.index, errors="coerce")
        wide = wide.loc[wide.index.notna()].apply(pd.to_numeric, errors="coerce").sort_index()
        wide.columns = wide.columns.map(str)
        if target_name.lower() == "speed":
            wide = wide.mask((wide < 0) | (wide > 160))
        quality = {"source": "uploaded CSV", "target": target_name, "sensors": wide.shape[1], "timestamps": len(wide)}
    else:
        if not Path(data_path).is_file():
            st.info("Upload a real dataset or set a path to datasets/traffic.csv. No sample/live data is generated.")
            st.stop()
        modified_ns = Path(data_path).stat().st_mtime_ns
        data, quality, coordinates = load_local_dataset(data_path, target_name, int(interval_minutes), modified_ns)
        wide = None
    if upload is not None:
        data = regularize_series(wide, int(interval_minutes))
except Exception as exc:
    st.error(f"Could not load traffic data: {exc}")
    st.stop()

st.info(f"Historical data from {data.index.min()} through {data.index.max()}. "
        f"The latest observation is not necessarily real-time.")
top = st.columns(4)
top[0].metric("Sensors", f"{data.shape[1]:,}")
top[1].metric("Complete timestamps", f"{len(data):,}")
top[2].metric("Latest observation", data.index.max().strftime("%Y-%m-%d %H:%M"))
top[3].metric("Measured target", target_name)
with st.expander("Data quality summary"):
    st.json(quality)

sensor = st.selectbox("Road / sensor", list(data.columns))
history = data[sensor].rename(target_name).dropna().to_frame()
left, right = st.columns([1.5, 1])
with left:
    st.subheader("Historical sensor trend")
    chart = px.line(history.tail(7 * 24 * 60 // int(interval_minutes)), title=f"Historical {target_name} — sensor {sensor}")
    st.plotly_chart(chart, width="stretch", alt=f"Historical {target_name} trend for sensor {sensor}")
with right:
    st.subheader("Recent observation")
    st.metric(f"Latest measured {target_name}", f"{history.iloc[-1, 0]:.2f}", help=f"Observed at {history.index[-1]}")
    if target_name.lower() == "speed":
        speed = float(history.iloc[-1, 0])
        congestion = "Severe" if speed < 15 else "High" if speed < 30 else "Moderate" if speed < 50 else "Low"
        st.metric("Speed-band congestion indicator", congestion)
        st.caption("Illustrative speed bands only; calibrate by road class and city. Not an official classification.")

if coordinates is not None:
    st.subheader("Sensor locations")
    coordinate_cols = {str(col).lower(): col for col in coordinates.columns}
    if {"latitude", "longitude"}.issubset(coordinate_cols):
        map_frame = coordinates.rename(columns={coordinate_cols["latitude"]: "latitude", coordinate_cols["longitude"]: "longitude"})
        map_frame["sensor_id"] = map_frame.get("sensor_id", map_frame.iloc[:, 0]).astype(str)
        map_frame["latitude"] = pd.to_numeric(map_frame["latitude"], errors="coerce")
        map_frame["longitude"] = pd.to_numeric(map_frame["longitude"], errors="coerce")
        map_frame = map_frame.merge(
            data.iloc[-1].rename(target_name).rename_axis("sensor_id").reset_index(),
            on="sensor_id",
            how="inner",
        )
        st.map(
            map_frame,
            latitude="latitude",
            longitude="longitude",
            color=target_name,
            alt=f"Map of sensor locations colored by latest measured {target_name}",
        )
    else:
        st.caption("Coordinates are not available in this data.")
else:
    st.caption("No GPS coordinates supplied; map display is unavailable.")

st.subheader("Forecasts")
model_dir = PROJECT_ROOT / "models"
try:
    bundle = load_prediction_bundle(model_dir)
    compatible = all(str(sensor_id) in {str(c) for c in data.columns} for sensor_id in bundle["metadata"]["sensor_ids"])
    if not compatible:
        st.warning("The trained model sensors do not match the currently loaded dataset.")
    else:
        forecast = predict_future(data, bundle)
        selected_forecast = forecast.loc[forecast["sensor_id"] == str(sensor)]
        if selected_forecast.empty:
            st.warning("No forecast is available for this sensor.")
        else:
            st.caption("Predictions are generated from the newest historical window; they are not live traffic readings.")
            if target_name.lower() == "speed":
                selected_forecast = selected_forecast.copy()
                selected_forecast["congestion_level"] = selected_forecast["predicted_value"].map(
                    lambda value: "Severe" if value < 15 else "High" if value < 30 else "Moderate" if value < 50 else "Low"
                )
                if selected_forecast["congestion_level"].isin(["High", "Severe"]).any():
                    st.warning("Forecast speed falls in a high/severe illustrative congestion band.")
            st.dataframe(
                selected_forecast[["horizon_minutes", "forecast_for", "target", "predicted_value", "model"] + (["congestion_level"] if "congestion_level" in selected_forecast else [])],
                hide_index=True,
                alt="Predicted traffic conditions by forecast horizon",
            )
except FileNotFoundError:
    st.info("No trained model bundle found. Run the training command in README.md to enable forecasts.")
except Exception as exc:
    st.error(f"Forecast unavailable: {exc}")

metrics_path = model_dir / "metrics.csv"
if metrics_path.is_file():
    st.subheader("Held-out model comparison")
    metrics_frame = pd.read_csv(metrics_path)
    if not metrics_frame.empty:
        metric_columns = [column for column in ("model", "horizon_minutes", "mae", "rmse", "mape_percent", "r2") if column in metrics_frame]
        st.dataframe(metrics_frame[metric_columns], hide_index=True, alt="Held-out forecast metrics by model and horizon")
        st.plotly_chart(
            px.bar(metrics_frame, x="horizon_minutes", y="mae", color="model", barmode="group", title="Held-out MAE by forecast horizon"),
            width="stretch",
            alt="Held-out mean absolute error comparison by model and horizon",
        )
        selected_path = model_dir / "selected_models.csv"
        if selected_path.is_file():
            selected_models = pd.read_csv(selected_path)
            if not selected_models.empty:
                plot_columns = st.columns(min(len(selected_models), 3))
                for position, (_, row) in enumerate(selected_models.iterrows()):
                    plot_path = model_dir / "plots" / f"{row['model']}_{int(row['horizon_minutes'])}m.png"
                    if plot_path.is_file():
                        plot_columns[position % len(plot_columns)].image(
                            str(plot_path),
                            caption=f"Best validation model, {int(row['horizon_minutes'])} minutes",
                            alt=f"Observed versus predicted held-out traffic for {int(row['horizon_minutes'])} minutes",
                            width="stretch",
                        )

if target_name.lower() in {"flow", "occupancy", "density"}:
    st.caption(f"Showing measured {target_name} values from the supplied data; semantics and units depend on the dataset.")
else:
    st.caption("Flow and occupancy are not displayed unless those fields are supplied and trained as targets.")
