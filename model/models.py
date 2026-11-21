import torch
import torch.nn as nn

from config import FFNN_PARAMS, LSTM_PARAMS


class FFNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        dims = (input_dim, *FFNN_PARAMS["hidden"])
        dropouts = FFNN_PARAMS["dropout"]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropouts[i]),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMForecaster(nn.Module):
    """(batch, seq_len, n_seq_features) + city one-hot -> t+6h temperature."""

    def __init__(self, n_seq_features: int, n_cities: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=LSTM_PARAMS["hidden"],
            num_layers=LSTM_PARAMS["layers"],
            batch_first=True,
            dropout=LSTM_PARAMS["dropout"],
        )
        self.head = nn.Sequential(
            nn.Linear(LSTM_PARAMS["hidden"] + n_cities, LSTM_PARAMS["head_hidden"]),
            nn.ReLU(),
            nn.Dropout(LSTM_PARAMS["dropout"]),
            nn.Linear(LSTM_PARAMS["head_hidden"], 1),
        )

    def forward(self, x_seq, city_onehot):
        out, _ = self.lstm(x_seq)          # (batch, seq, hidden)
        last = out[:, -1, :]               # hidden state at hour t
        return self.head(
            torch.cat([last, city_onehot], dim=1)
        ).squeeze(-1) 