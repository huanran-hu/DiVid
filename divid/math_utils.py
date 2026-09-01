"""Small numerical utilities used by the six DiVid dimensions."""

from __future__ import annotations

import math

import numpy as np


def l2_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), eps)


def upper_triangle_mean(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[0] < 2:
        return 0.0
    rows, cols = np.triu_indices(values.shape[0], k=1)
    return float(values[rows, cols].mean())


def pairwise_cosine_distance(features: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    normalized = l2_normalize(features, eps=eps)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    return distance, upper_triangle_mean(distance)


def psd_clip(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return 0.5 * (clipped + clipped.T)


def normalize_kernel_diagonal(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    diagonal = np.maximum(np.diag(values), eps)
    inverse = 1.0 / np.sqrt(diagonal)
    normalized = inverse[:, None] * values * inverse[None, :]
    normalized = 0.5 * (normalized + normalized.T)
    np.fill_diagonal(normalized, 1.0)
    return normalized


def centered_cosine_kernel(features: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(features, dtype=np.float64) - np.asarray(features, dtype=np.float64).mean(axis=0)
    normalized = l2_normalize(centered, eps=eps)
    kernel = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(kernel, 1.0)
    return normalized, kernel


def vendi_score_from_kernel(kernel: np.ndarray, eps: float = 1e-8) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(psd_clip(kernel)), 0.0)
    total = float(eigenvalues.sum())
    if total <= eps:
        return 1.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > eps]
    return float(math.exp(float(-(probabilities * np.log(probabilities)).sum())))


def rbf_metrics(features: np.ndarray, scales: tuple[float, ...], sigma: float, eps: float = 1e-8) -> dict[str, np.ndarray | float]:
    values = np.asarray(features, dtype=np.float64)
    scale_array = np.asarray(scales, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(scale_array):
        raise ValueError("Feature dimension and scale count do not match.")
    standardized = values / np.maximum(scale_array[None, :], eps)
    differences = standardized[:, None, :] - standardized[None, :, :]
    raw_distance = np.linalg.norm(differences, axis=-1)
    similarity = np.exp(-(raw_distance * raw_distance) / (2.0 * max(float(sigma), eps) ** 2))
    np.fill_diagonal(similarity, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return {
        "similarity": similarity,
        "distance": distance,
        "mpd": upper_triangle_mean(distance),
        "raw_distance": raw_distance,
        "raw_mpd": upper_triangle_mean(raw_distance),
        "vendi": vendi_score_from_kernel(similarity, eps=eps),
        "standardized": standardized,
    }
