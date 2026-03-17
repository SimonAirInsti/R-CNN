"""
Recurrent Convolutional Neural Network (RCNN) for protein antigenicity prediction.

Architecture:
    1. Multi-scale 1D convolutions → local feature extraction from ESM-2 per-residue embeddings
    2. Bidirectional LSTM → sequential dependency modeling
    3. Global pooling (average + max) → fixed-size sequence representation
    4. FC classifier → binary antigenicity prediction
"""

import torch
import torch.nn as nn


class RCNN(nn.Module):
    """
    RCNN for sequence-level binary classification from per-residue ESM-2 embeddings.

    Args:
        input_dim:    Dimension per residue (e.g. 1280 for ESM-2 t33)
        num_filters:  Filters per convolutional branch
        kernel_sizes: Kernel sizes for multi-scale convolutions
        rnn_hidden:   LSTM hidden size (per direction)
        rnn_layers:   Number of stacked LSTM layers
        fc_hidden:    Hidden size of the classifier head
        dropout:      Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_filters: int = 128,
        kernel_sizes: list = None,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        # --- Convolutional feature extraction (multi-scale) ---
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, num_filters, ks, padding=ks // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for ks in kernel_sizes
            ]
        )

        conv_out_dim = num_filters * len(kernel_sizes)

        # --- Bidirectional LSTM ---
        self.rnn = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # avg_pool ‖ max_pool  ×  bidirectional
        pool_dim = rnn_hidden * 2 * 2

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       Padded embeddings  [batch, max_seq_len, embed_dim]
            lengths: Actual lengths     [batch]
        Returns:
            logits   [batch]
        """
        # --- Conv1D expects [B, C, L] ---
        x_t = x.permute(0, 2, 1)
        conv_outs = [block(x_t) for block in self.conv_blocks]
        x_t = torch.cat(conv_outs, dim=1)          # [B, conv_out_dim, L]
        x_seq = x_t.permute(0, 2, 1)               # [B, L, conv_out_dim]

        # --- Packed BiLSTM ---
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # --- Masked global pooling ---
        lengths_d = lengths.to(rnn_out.device).float()
        mask = (
            torch.arange(rnn_out.size(1), device=rnn_out.device).unsqueeze(0)
            < lengths_d.unsqueeze(1).long()
        ).unsqueeze(2)                              # [B, L, 1]

        avg_pool = (rnn_out * mask.float()).sum(1) / lengths_d.unsqueeze(1)

        neg_inf = torch.full_like(rnn_out, float("-inf"))
        rnn_masked = torch.where(mask.expand_as(rnn_out), rnn_out, neg_inf)
        max_pool = rnn_masked.max(dim=1).values

        pooled = torch.cat([avg_pool, max_pool], dim=1)   # [B, pool_dim]
        return self.classifier(pooled).squeeze(-1)