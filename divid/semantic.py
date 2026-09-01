"""Prompt-conditioned Semantic diversity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_CONFIG, EvalConfig
from .math_utils import l2_normalize, normalize_kernel_diagonal, psd_clip, upper_triangle_mean, vendi_score_from_kernel
from .video import sample_frames


def _conditioned_kernel(image_features: np.ndarray, text_feature: np.ndarray, eps: float) -> np.ndarray:
    images = l2_normalize(image_features, eps=eps)
    text = l2_normalize(np.asarray(text_feature)[None, :], eps=eps)[0]
    image_kernel = images @ images.T
    prompt_similarity = images @ text
    conditioned = image_kernel - np.outer(prompt_similarity, prompt_similarity)
    return normalize_kernel_diagonal(psd_clip(conditioned), eps=eps)


@torch.inference_mode()
def compute_semantic(
    video_paths: list[str | Path],
    prompt: str,
    *,
    device: str,
    config: EvalConfig = DEFAULT_CONFIG,
    bundle=None,
) -> dict:
    if len(video_paths) < 2:
        raise ValueError("Semantic diversity requires at least two videos.")
    if bundle is None:
        from .models import ModelBundle

        with ModelBundle(device=device, config=config) as owned:
            return compute_semantic(video_paths, prompt, device=device, config=config, bundle=owned)

    model, preprocess, tokenizer = bundle.semantic.model, bundle.semantic.preprocess, bundle.semantic.tokenizer
    video_features = []
    for path in video_paths:
        frames = sample_frames(path, config.clip_frames)
        batch = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(device)
        frame_features = torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1)
        video_features.append(torch.nn.functional.normalize(frame_features.mean(0), dim=0).cpu().numpy())

    text = tokenizer([prompt]).to(device)
    text_feature = torch.nn.functional.normalize(model.encode_text(text).float(), dim=-1)[0].cpu().numpy()
    kernel = _conditioned_kernel(np.asarray(video_features), text_feature, config.eps)
    distance = np.clip(1.0 - kernel, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    return {
        "semantic_mpd": upper_triangle_mean(distance),
        "semantic_vendi": vendi_score_from_kernel(kernel, eps=config.eps),
        "semantic_similarity": kernel.tolist(),
        "semantic_distance": distance.tolist(),
        "semantic_valid_videos": len(video_paths),
    }
