import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocessing import load_traffic_data, regularize_series
from src.feature_engineering import build_model_features, fit_scale_train_only, make_sequences
from src.graph_construction import adjacency_from_coordinates, adjacency_from_training_speeds
from src.model_evaluation import calculate_metrics


class TrafficPipelineTests(unittest.TestCase):
    def test_long_csv_is_pivoted_and_duplicate_sensor_rows_keep_last(self):
        data = pd.DataFrame({
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:05"],
            "sensor_id": ["a", "a", "a"],
            "speed": [20, 25, 30],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.csv"
            data.to_csv(path, index=False)
            frame, summary, _ = load_traffic_data(path)
        self.assertEqual(frame.iloc[0, 0], 25)
        self.assertEqual(summary["sensors"], 1)

    def test_sequence_target_is_exactly_horizon_steps_after_last_input(self):
        index = pd.date_range("2024-01-01", periods=12, freq="5min")
        values = np.arange(12, dtype=np.float32).reshape(-1, 1)
        sequences = make_sequences(values, index, lookback=3, horizon_steps=3)
        self.assertEqual(sequences.x[0, :, 0, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(sequences.y[0, 0], 5.0)
        self.assertEqual(sequences.target_times[0], index[5])

    def test_calendar_features_are_cyclical_and_match_sensor_dimensions(self):
        index = pd.date_range("2024-01-01", periods=4, freq="6h")
        features = build_model_features(np.ones((4, 2), dtype=np.float32), index)
        self.assertEqual(features.shape, (4, 2, 7))
        np.testing.assert_array_equal(features[:, 0, 1:], features[:, 1, 1:])

    def test_scaler_is_fitted_only_before_training_boundary(self):
        index = pd.date_range("2024-01-01", periods=10, freq="5min")
        frame = pd.DataFrame({"a": np.r_[np.arange(8), 1000, 1000]}, index=index)
        _, scaler = fit_scale_train_only(frame, index[8])
        self.assertEqual(scaler.mean_[0], 3.5)

    def test_regularizer_forward_fills_only_configured_short_gap(self):
        index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:15"])
        frame = pd.DataFrame({"a": [10.0, 40.0]}, index=index)
        regular = regularize_series(frame, interval_minutes=5, fill_limit=1)
        self.assertEqual(regular["a"].tolist(), [10.0, 10.0, 40.0])

    def test_graph_is_symmetric_and_metrics_are_finite(self):
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
        adjacency = adjacency_from_training_speeds(frame)
        np.testing.assert_allclose(adjacency, adjacency.T)
        metrics = calculate_metrics(np.array([[1.0, 2.0]]), np.array([[1.0, 3.0]]))
        self.assertEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.5))

    def test_coordinate_graph_matches_sensor_ids_as_strings(self):
        coordinates = pd.DataFrame({
            "sensor_id": [101, 202],
            "latitude": [34.0, 34.01],
            "longitude": [-118.0, -118.01],
        })
        adjacency = adjacency_from_coordinates(["101", "202"], coordinates)
        self.assertEqual(adjacency.shape, (2, 2))
        self.assertGreater(adjacency[0, 1], 0)


if __name__ == "__main__":
    unittest.main()
