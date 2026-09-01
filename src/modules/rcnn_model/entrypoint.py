from __future__ import annotations

import torch
import torch.nn as nn


class RCNN(nn.Module):
    """Base RCNN for sequence-level antigenicity prediction.

    The project configuration lives in config/rcnn_config.yaml and should be the
    canonical source for architecture hyperparameters. These constructor defaults
    exist only as a minimal fallback for compatibility and safety, not as the
    real configuration of the experimental setup.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_filters: int = 128,
        kernel_sizes: list[int] | None = None,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

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

        self.rnn = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        pool_dim = rnn_hidden * 2 * 2

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(0, 2, 1)
        conv_outs = [block(x_t) for block in self.conv_blocks]
        x_t = torch.cat(conv_outs, dim=1)
        x_seq = x_t.permute(0, 2, 1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        lengths_d = lengths.to(rnn_out.device).float()
        mask = (
            torch.arange(rnn_out.size(1), device=rnn_out.device).unsqueeze(0)
            < lengths_d.unsqueeze(1).long()
        ).unsqueeze(2)

        avg_pool = (rnn_out * mask.float()).sum(1) / lengths_d.unsqueeze(1)

        neg_inf = torch.full_like(rnn_out, float("-inf"))
        rnn_masked = torch.where(mask.expand_as(rnn_out), rnn_out, neg_inf)
        max_pool = rnn_masked.max(dim=1).values

        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(config: dict) -> RCNN:
    """Build the model strictly from the project YAML config.

    The canonical configuration is read from config/rcnn_config.yaml and passed
    here as a dictionary. The constructor defaults are only a defensive fallback
    for isolated usage outside the project configuration flow.
    """
    model_cfg = config.get("model", {}).get("base", {})
    return RCNN(
        input_dim=model_cfg.get("input_dim", 1280),
        num_filters=model_cfg.get("num_filters", 128),
        kernel_sizes=model_cfg.get("kernel_sizes", [3, 5, 7]),
        rnn_hidden=model_cfg.get("rnn_hidden", 128),
        rnn_layers=model_cfg.get("rnn_layers", 2),
        fc_hidden=model_cfg.get("fc_hidden", 128),
        dropout=model_cfg.get("dropout", 0.3),
    )
