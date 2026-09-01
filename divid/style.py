"""Global appearance/style diversity based on InceptionV3 pool3 features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from .config import DEFAULT_CONFIG, EvalConfig
from .math_utils import pairwise_cosine_distance, vendi_score_from_kernel
from .video import sample_frames


def inception_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.inference_mode()
def compute_style(
    video_paths: list[str | Path],
    *,
    device: str,
    config: EvalConfig = DEFAULT_CONFIG,
    bundle=None,
) -> dict:
    if len(video_paths) < 2:
        raise ValueError("Style diversity requires at least two videos.")
    if bundle is None:
        from .models import ModelBundle

        with ModelBundle(device=device, config=config) as owned:
            return compute_style(video_paths, device=device, config=config, bundle=owned)

    model, transform = bundle.style.model, bundle.style.transform
    features = []
    for path in video_paths:
        frames = sample_frames(path, config.inception_frames)
        batch = torch.stack([transform(frame) for frame in frames]).to(device)
        frame_features = torch.nn.functional.normalize(model(batch).float(), dim=-1)
        features.append(torch.nn.functional.normalize(frame_features.mean(0), dim=0).cpu().numpy())
    distance, mpd = pairwise_cosine_distance(np.asarray(features), eps=config.eps)
    similarity = 1.0 - distance
    np.fill_diagonal(similarity, 1.0)
    return {
        "style_mpd": mpd,
        "style_vendi": vendi_score_from_kernel(similarity, eps=config.eps),
        "style_similarity": similarity.tolist(),
        "style_distance": distance.tolist(),
        "style_valid_videos": len(video_paths),
    }
