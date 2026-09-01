"""Public API for evaluating one prompt-conditioned video set."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import DEFAULT_CONFIG, EvalConfig
from .geometry import compute_motion_camera
from .masks import compute_subject_mask_tracks
from .semantic import compute_semantic
from .style import compute_style
from .subject_scene import compute_subject_scene
from .models import ModelBundle

DIMENSIONS = ("semantic", "style", "subject", "scene", "motion", "camera")


def evaluate(
    videos: Iterable[str | Path],
    prompt: str,
    *,
    subjects: Iterable[str] | None = None,
    dimensions: Iterable[str] | None = None,
    device: str = "cuda",
    config: EvalConfig = DEFAULT_CONFIG,
) -> dict:
    """Return six-dimensional diversity scores for one video set.

    ``videos`` must contain at least two videos generated from ``prompt``.
    For Subject, Scene, Motion, and Camera, ``subjects`` should contain the
    foreground entities to localize, for example ``["a dog", "a ball"]``.
    """

    paths = [Path(path) for path in videos]
    if len(paths) < 2:
        raise ValueError("At least two videos are required.")
    requested = list(dimensions) if dimensions is not None else list(DIMENSIONS)
    requested = [str(item).lower() for item in requested]
    unknown = sorted(set(requested) - set(DIMENSIONS))
    if unknown:
        raise ValueError(f"Unknown dimensions: {', '.join(unknown)}")
    result = {"prompt": prompt, "video_count": len(paths), "dimensions": requested}

    with ModelBundle(device=device, config=config) as bundle:
        mask_tracks = None
        mask_dimensions = {"subject", "scene", "motion", "camera"} & set(requested)
        if mask_dimensions:
            queries = tuple(subjects or (prompt,))
            mask_tracks = compute_subject_mask_tracks(paths, queries, device=device, config=config, bundle=bundle)
        if "semantic" in requested:
            result.update(compute_semantic(paths, prompt, device=device, config=config, bundle=bundle))
        if "style" in requested:
            result.update(compute_style(paths, device=device, config=config, bundle=bundle))
        if {"subject", "scene"} & set(requested):
            result.update(compute_subject_scene(paths, device=device, config=config, bundle=bundle, mask_tracks=mask_tracks))
        if {"motion", "camera"} & set(requested):
            result.update(compute_motion_camera(paths, device=device, config=config, bundle=bundle, mask_tracks=mask_tracks))
    return result
