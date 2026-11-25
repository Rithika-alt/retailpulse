"""LSTM sequence forecaster: 28-day history + known-at-target covariates -> next-day sales."""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_seq_features: int,
        n_known_features: int,
        n_items: int,
        n_stores: int,
        hidden: int = 64,
        layers: int = 2,
        emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.store_emb = nn.Embedding(n_stores, emb_dim)
        self.lstm = nn.LSTM(
            n_seq_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_known_features + 2 * emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, known, item, store):
        out, _ = self.lstm(x)  # (B, W, H)
        h = out[:, -1]  # state after day D-1
        z = torch.cat([h, known, self.item_emb(item), self.store_emb(store)], dim=1)
        return self.head(z).squeeze(-1)
