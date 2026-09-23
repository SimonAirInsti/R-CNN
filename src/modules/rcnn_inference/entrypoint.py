from __future__ import annotations

import os

import torch

from src.modules.rcnn_model.entrypoint import RCNN


def load_model(model_path: str, device: torch.device, model_class: type[RCNN] = RCNN) -> RCNN:
    model = model_class()
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_sequence(model: RCNN, embeddings: torch.Tensor, length: int, device: torch.device) -> float:
    x = embeddings.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x, lengths)
        prob = torch.sigmoid(logits).item()
    return float(prob)
