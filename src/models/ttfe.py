"""
Transformer Temporal Feature Extractor (TTFE).

Processes rolling windows of price vectors through a transformer encoder
to produce compressed temporal feature vectors for the SAC policy.

Reference: Li et al. (2024), arXiv:2402.19110
"""

import torch
import torch.nn as nn


class TTFE(nn.Module):
    """
    Transformer Temporal Feature Extractor.

    Input:  (batch, seq_len, n_prices) price history
    Output: (batch, d_model) temporal feature vector
    """

    def __init__(
        self,
        n_prices: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        seq_len: int = 32,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_prices = n_prices
        self.d_model = d_model
        self.seq_len = seq_len

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)

        # Input projection: n_prices -> d_model
        self.input_proj = nn.Linear(n_prices, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, n_prices)

        Returns
        -------
        Tensor of shape (batch, d_model)
        """
        # Project input prices to d_model dimension
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add learnable positional embeddings
        h = h + self.pos_embedding  # broadcasts over batch

        # Transformer encoder
        h = self.transformer(h)  # (batch, seq_len, d_model)

        # Take last timestep as output
        return h[:, -1, :]  # (batch, d_model)
