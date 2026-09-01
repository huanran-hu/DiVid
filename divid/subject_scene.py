"""Foreground Subject and background Scene diversity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from .config import DEFAULT_CONFIG, EvalConfig
from .masks import SubjectMaskTracks, is_valid_mask, resize_mask, compute_subject_mask_tracks
from .math_utils import l2_normalize, pairwise_cosine_distance, vendi_score_from_kernel


def _device(value, device: str):
    return value.to(device) if hasattr(value, "to") else value


def _embed(image: Image.Image, bundle, device: str) -> np.ndarray:
    inputs = _device(bundle.dinov2_processor(images=image, return_tensors="pt"), device)
    with torch.inference_mode():
        output = bundle.dinov2_model(**inputs).last_hidden_state
    cls = output[:, 0, :]
    patch_mean = output[:, 1:, :].mean(dim=1) if output.shape[1] > 1 else cls
    feature = torch.nn.functional.normalize(torch.cat([cls, patch_mean], dim=-1), dim=-1)
    return feature[0].cpu().numpy().astype(np.float64)


def _mean_embedding(features: list[np.ndarray], eps: float) -> np.ndarray:
    if not features:
        raise ValueError("No valid frames were available for embedding.")
    return l2_normalize(np.mean(np.asarray(features, dtype=np.float64), axis=0, keepdims=True), eps=eps)[0]


def _subject_image(frame: Image.Image, mask: np.ndarray) -> Image.Image:
    image = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
    mask = resize_mask(mask, frame.size)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return frame.convert("RGB")
    fill = np.round(image.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    image[~mask] = fill
    return Image.fromarray(image[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1])


def _scene_image(frame: Image.Image, mask: np.ndarray) -> Image.Image:
    source = frame.convert("RGB")
    mask = resize_mask(mask, source.size)
    blurred = source.filter(ImageFilter.GaussianBlur(radius=12))
    return Image.composite(blurred, source, Image.fromarray(mask.astype(np.uint8) * 255, mode="L"))


def _metrics(features: np.ndarray, eps: float) -> dict:
    normalized = l2_normalize(features, eps=eps)
    distance, mpd = pairwise_cosine_distance(normalized, eps=eps)
    similarity = 1.0 - distance
    np.fill_diagonal(similarity, 1.0)
    return {
        "mpd": mpd,
        "vendi": vendi_score_from_kernel(similarity, eps=eps),
        "similarity": similarity.tolist(),
        "distance": distance.tolist(),
    }


def compute_subject_scene(
    video_paths: list[str | Path],
    *,
    device: str,
    config: EvalConfig = DEFAULT_CONFIG,
    bundle=None,
    mask_tracks: SubjectMaskTracks | None = None,
) -> dict:
    if bundle is None:
        from .models import ModelBundle

        with ModelBundle(device=device, config=config) as owned:
            return compute_subject_scene(video_paths, device=device, config=config, bundle=owned, mask_tracks=mask_tracks)
    if mask_tracks is None or mask_tracks.status != "ok":
        raise ValueError("Valid subject masks are required for Subject and Scene diversity.")
    if len(mask_tracks.valid_tracks) < 2:
        raise ValueError("At least two valid subject-mask videos are required.")

    model_bundle = bundle.subject_scene
    subject_features_by_query: list[np.ndarray] = []
    subject_weights: list[float] = []
    scene_features: list[np.ndarray] = []
    for track in mask_tracks.valid_tracks:
        query_features: list[np.ndarray] = []
        query_areas: list[float] = []
        for subject in mask_tracks.subjects:
            frame_features = []
            for frame, mask in zip(track.frames, track.subject_masks[subject]):
                if is_valid_mask(mask, config):
                    query_areas.append(float(np.asarray(mask).mean()))
                    frame_features.append(_embed(_subject_image(frame, mask), model_bundle, device))
            if frame_features:
                query_features.append(_mean_embedding(frame_features, config.eps))
        if not query_features:
            raise ValueError(f"No valid subject features for {track.video_path}.")
        subject_features_by_query.append(np.mean(query_features, axis=0))
        subject_weights.append(float(np.mean(query_areas)) if query_areas else 1.0)

        frame_features = []
        for frame, mask in zip(track.frames, track.union_masks):
            if is_valid_mask(mask, config):
                frame_features.append(_embed(_scene_image(frame, mask), model_bundle, device))
        scene_features.append(_mean_embedding(frame_features, config.eps))

    subject = _metrics(np.asarray(subject_features_by_query), config.eps)
    scene = _metrics(np.asarray(scene_features), config.eps)
    return {
        "subject_mpd": subject["mpd"],
        "subject_vendi": subject["vendi"],
        "subject_similarity": subject["similarity"],
        "subject_distance": subject["distance"],
        "subject_valid_videos": len(mask_tracks.valid_tracks),
        "scene_mpd": scene["mpd"],
        "scene_vendi": scene["vendi"],
        "scene_similarity": scene["similarity"],
        "scene_distance": scene["distance"],
        "scene_valid_videos": len(mask_tracks.valid_tracks),
    }
