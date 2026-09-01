"""Small flow helpers shared by Motion v4.1 and Camera v3."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def raft_flow(model, transform, frame_a: np.ndarray, frame_b: np.ndarray, *, device: str) -> np.ndarray:
    image_a, image_b = transform(Image.fromarray(frame_a), Image.fromarray(frame_b))
    image_a = image_a.unsqueeze(0)
    image_b = image_b.unsqueeze(0)
    _, _, height, width = image_a.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        image_a = F.pad(image_a, (0, pad_w, 0, pad_h), mode="replicate")
        image_b = F.pad(image_b, (0, pad_w, 0, pad_h), mode="replicate")
    prediction = model(image_a.to(device), image_b.to(device))[-1]
    return prediction.squeeze(0).detach().cpu().numpy()[:, :height, :width]


def affine_flow(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    transformed = points @ matrix[:, :2].T + matrix[:, 2]
    flow = (transformed - points).reshape(height, width, 2)
    return flow.transpose(2, 0, 1).astype(np.float32)


def normalize_probability(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(array.sum())
    return array / total if total > eps else np.zeros_like(array)


def direction_histogram(directions: np.ndarray, weights: np.ndarray, bins: int = 8) -> np.ndarray:
    if directions.size == 0:
        return np.zeros(bins, dtype=np.float64)
    histogram, _ = np.histogram(directions, bins=bins, range=(-math.pi, math.pi), weights=weights)
    return normalize_probability(histogram)


def motion_activity_weight(ratio: float, minimum: float, full: float, eps: float = 1e-8) -> float:
    value = max(float(ratio), 0.0)
    if value < float(minimum):
        return 0.0
    return float(np.clip(value / max(float(full), eps), 0.0, 1.0))
