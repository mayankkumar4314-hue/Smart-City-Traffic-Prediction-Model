# Smart City Traffic Flow Prediction

A reproducible research project for short-term traffic forecasting from historical, timestamped sensor observations. It includes a preprocessing pipeline, chronological evaluation, baseline and spatio-temporal models, an EDA notebook, and a Streamlit dashboard.

> **Data honesty:** the project does not ship traffic observations, model weights, performance claims, or a live traffic feed. METR-LA primarily provides measured traffic speed; flow and occupancy are not inferred or presented as measurements when they are absent. The dashboard identifies the timestamp of the newest observation and labels forecasts as historical-data-based.

## What is included

- CSV and HDF5 ingestion for long-form sensor data and wide sensor-by-time data.
- Data quality checks, speed-range filtering, resampling, and bounded forward filling.
- Chronological train/validation/test splits and train-only feature scaling.
- Cyclical hour-of-day, day-of-week, and annual time features alongside sensor histories.
- Linear regression, random forest, support vector regression, optional XGBoost, LSTM, GRU, CNN-LSTM, graph-convolutional LSTM, and STGCN forecasters.
- 15/30/60-minute forecasts (horizons are converted to the configured data interval).
- MAE, RMSE, MAPE, and R² reporting, model selection on validation MAE, and held-out test evaluation.
- Streamlit dashboard for historical observations, map visualization when coordinates exist, and forecasts when trained models exist.

## Project layout

```text
smart-city-traffic-prediction/
├── datasets/                 # User-provided data; not committed
├── notebooks/eda.ipynb       # Reproducible EDA notebook
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── graph_construction.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── models.py
│   ├── traffic_prediction.py
│   └── dashboard/app.py
├── models/                   # Generated model artifacts; not committed
├── visualizations/           # Optional exported analysis figures
├── docs/architecture.md
├── requirements.txt
└── README.md
```

## Dataset selection and collection

METR-LA and PEMS-BAY are established public traffic-sensor benchmarks. METR-LA is a good starting point for network-aware speed prediction because it contains regularly sampled speed observations from many sensors. Its standard data release does **not** provide all the fields in the project objective: do not interpret a speed-only record as measured flow, occupancy, weather, or GPS.

1. Follow the DCRNN benchmark [data preparation instructions](https://github.com/liyaguang/DCRNN#data-preparation). The repository links the METR-LA and PEMS-BAY HDF5 files on [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) (and Baidu), with filenames `metr-la.h5` and `pems-bay.h5`. Review the source license/terms and verify the download before use.
2. Put the chosen file under `datasets/` (for example `datasets/METR-LA.h5`). The HDF5 loader expects the benchmark `df` key, with timestamps in rows and sensor IDs in columns. Large data and derived model files are intentionally excluded from version control.
3. Optional sensor coordinates are available in the DCRNN repository's [METR-LA locations CSV](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv) and [PEMS-BAY locations CSV](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations_bay.csv). Supply a compatible CSV (`sensor_id,latitude,longitude`) with `--coordinates-csv`, or include those fields in a long-form traffic CSV.
4. For a non-benchmark CSV, use either long form (`timestamp,sensor_id,speed`) or wide form (`timestamp,sensor_1,sensor_2,...`). Extra measured numeric columns such as `flow` and `occupancy` are preserved by the loader, but the current model pipeline selects one target at a time.

The loader removes duplicate timestamp/sensor records (keeping the last), sorts time, rejects physically implausible speed values, resamples to a regular interval, and fills only short gaps forward. Longer gaps remain missing and are excluded from complete training sequences. Review the data-quality summary before trusting a model; this simple pipeline is not a substitute for dataset-specific sensor calibration.

## Installation

Python 3.10+ is recommended. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can be platform/CUDA dependent. If the default package index does not provide a suitable wheel, install the matching CPU or CUDA build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) and then install the remaining requirements.

## Run preprocessing and training

Example for a long-form CSV with 5-minute speed samples:

```powershell
python -m src.model_training --data datasets/traffic.csv --timestamp-col timestamp --sensor-col sensor_id --target speed --interval-minutes 5 --output-dir models
```

For METR-LA HDF5 (`df` key is the default; rows are timestamps and columns are sensor IDs):

```powershell
python -m src.model_training --data datasets/METR-LA.h5 --target speed --interval-minutes 5 --output-dir models
```

For a separate coordinate file, add `--coordinates-csv datasets/sensor_locations.csv` (columns: `sensor_id,latitude,longitude`).

The default model set is a manageable subset (`linear,random_forest,lstm,gcn_lstm,stgcn`). To run all implemented baselines and neural models, pass e.g. `--models linear,random_forest,svr,xgboost,lstm,gru,cnn_lstm,gcn_lstm,stgcn`; XGBoost requires its optional package. Training saves per-horizon model files, a train-fitted scaler, the sensor order, metrics, and prediction plots in the output directory. Model selection uses validation MAE; the test set is reserved for final reporting.

The split is chronological (70% train, 15% validation, 15% test by target timestamp). Sequences use only observations at or before their forecast origin; scaling statistics are fitted on training observations only. Avoid tuning model choices against test metrics.

## Explore and launch the dashboard

Open `notebooks/eda.ipynb` in Jupyter and set `DATA_PATH` to the local dataset path.

```powershell
python -m streamlit run src/dashboard/app.py
```

The dashboard can read `datasets/traffic.csv` by default or accept a CSV upload. Before training, it can display historical sensor observations and measured metrics. Forecasts appear only when compatible trained artifacts are present in `models/`. Without sensor coordinates it displays a sensor selector and time series rather than fabricating a map. Without a live-data integration it is **not** a real-time traffic service: "current" means the latest observation in the supplied historical file.

## Models and evaluation

The baselines learn from flattened recent target windows. LSTM and GRU learn per-sensor temporal representations. CNN-LSTM extracts local temporal patterns with a 1-D convolution before recurrent processing. GCN-LSTM performs a normalized adjacency aggregation over sensors at each historical time step before temporal modeling. STGCN combines gated temporal convolutions with a spatial graph-convolution layer. For a supplied coordinate file, the graph builder uses a distance-based Gaussian kernel; otherwise a correlation graph can be estimated from training data only. Inputs also include cyclical hour, weekday, and annual time features so models can distinguish recurring calendar periods without using future observations.

At prediction origin \(t\), a model maps the preceding window \(X_{t-L+1:t}\) to sensor-level speed at \(t+h\). The graph layer computes a learned transform of the local sensor signal and its normalized neighbors. Validation MAE picks a model separately for each horizon. Test results are reported by model and horizon, not described as expected future performance.

MAE and RMSE are in the target's units (typically speed); MAPE excludes near-zero actual values and is reported as a percentage. R² is computed over the flattened sensor-time targets. These metrics should be interpreted alongside sensor coverage, missingness, and any dataset shift.

## Limitations and extensions

- A speed-only benchmark cannot support measured flow or occupancy predictions without additional labeled data.
- Benchmark sensor graphs/coordinates and sampling intervals vary by release. Confirm sensor order and metadata before using a supplied adjacency graph.
- The included graph-LSTM and compact STGCN are research baselines, not reproductions of DCRNN or production traffic-control systems.
- Weather, incidents, road closures, signal state, streaming ingestion, and online model monitoring are not implemented.
- Congestion labels in the dashboard are transparent speed bands, not official city classifications; configure thresholds for the target city and road type.
- Traffic signal optimization, emergency vehicle routing, and control actions are future extensions and are not implemented or validated here.

## Smart-city use cases

Forecasts can support traffic-operator situational awareness, bus arrival estimates, candidate corridor planning, and offline analysis of congestion patterns. In a validated deployment they may inform signal-management or emergency-route systems, but those systems require operational constraints, safety validation, and human oversight. Reduced emissions are a potential downstream benefit to measure, not a result claimed by this project.

See [docs/architecture.md](docs/architecture.md) for the architecture diagram and detailed roadmap.
