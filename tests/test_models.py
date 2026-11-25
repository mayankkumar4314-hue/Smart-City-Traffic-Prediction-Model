import unittest

import numpy as np
import torch

from src.models import build_neural_model


class NeuralModelTests(unittest.TestCase):
    def test_all_neural_models_return_one_forecast_per_sensor(self):
        adjacency = np.eye(3, dtype=np.float32)
        inputs = torch.ones((2, 4, 3, 7), dtype=torch.float32)
        for name in ("lstm", "gru", "cnn_lstm", "gcn_lstm", "stgcn"):
            with self.subTest(model=name):
                model = build_neural_model(name, adjacency, input_size=7)
                self.assertEqual(tuple(model(inputs).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
