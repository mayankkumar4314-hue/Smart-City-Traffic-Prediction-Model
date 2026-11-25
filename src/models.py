"""Baseline estimators and compact temporal/graph neural forecasting models."""

from __future__ import annotations

import torch
from torch import nn


class TemporalForecaster(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        if kind == "cnn_lstm":
            self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            recurrent = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = recurrent(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch, steps, sensors, features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * sensors, steps, features)
        if self.kind == "cnn_lstm":
            x = self.conv(x.transpose(1, 2)).transpose(1, 2).relu()
        output, _ = self.rnn(x)
        return self.head(output[:, -1]).reshape(batch, sensors)


class GraphTemporalForecaster(nn.Module):
    def __init__(self, adjacency: torch.Tensor, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.graph_projection = nn.Linear(2 * input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch, steps, sensors, _features = x.shape
        neighbors = torch.einsum("ij,btjf->btif", self.adjacency, x)
        graph_features = torch.cat((x, neighbors), dim=-1)
        graph_features = self.graph_projection(graph_features).relu()
        temporal = graph_features.permute(0, 2, 1, 3).reshape(batch * sensors, steps, -1)
        output, _ = self.rnn(temporal)
        return self.head(output[:, -1]).reshape(batch, sensors)


class GatedTemporalConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels,
            output_channels * 2,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

    def forward(self, x):
        value, gate = self.projection(x).chunk(2, dim=1)
        return torch.tanh(value) * torch.sigmoid(gate)


class SpatialGraphConv(nn.Module):
    def __init__(self, adjacency: torch.Tensor, input_channels: int, output_channels: int):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        neighbors = torch.einsum("ij,bctj->bcti", self.adjacency, x)
        return torch.relu(self.projection(x + neighbors))


class STGCNForecaster(nn.Module):
    """Two temporal gated convolutions around a spatial graph convolution."""

    def __init__(self, adjacency: torch.Tensor, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.temporal_in = GatedTemporalConv(input_size, hidden_size)
        self.spatial = SpatialGraphConv(adjacency, hidden_size, hidden_size)
        self.temporal_out = GatedTemporalConv(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_in(x)
        x = self.spatial(x)
        x = self.temporal_out(x)
        return self.head(x[:, :, -1, :].transpose(1, 2)).squeeze(-1)


def build_neural_model(name: str, adjacency, input_size: int, hidden_size: int = 32):
    if name in {"lstm", "gru", "cnn_lstm"}:
        return TemporalForecaster(name, input_size, hidden_size)
    if name == "gcn_lstm":
        return GraphTemporalForecaster(torch.as_tensor(adjacency, dtype=torch.float32), input_size, hidden_size)
    if name == "stgcn":
        return STGCNForecaster(torch.as_tensor(adjacency, dtype=torch.float32), input_size, hidden_size)
    raise ValueError(f"Unknown neural model: {name}")
