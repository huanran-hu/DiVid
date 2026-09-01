"""Validated defaults for the six-dimensional DiVid evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalConfig:
    frame_dir_fps: float = 10.0
    eps: float = 1e-8

    clip_model: str = "ViT-L-14-quickgelu"
    # Use the local checkpoint when available; the evaluator falls back to the
    # public OpenCLIP OpenAI weights if it is absent.
    clip_pretrained: str = "models/open_clip_model.safetensors"
    clip_frames: int = 8

    inception_weights: str = "models/inception_v3_google-0cc3c7bd.pth"
    inception_frames: int = 8

    subject_scene_grounding_model: str = "IDEA-Research/grounding-dino-base"
    subject_scene_sam2_model: str = "facebook/sam2.1-hiera-large"
    subject_scene_dinov2_model: str = "facebook/dinov2-large"
    subject_scene_grounding_box_threshold: float = 0.30
    subject_scene_grounding_text_threshold: float = 0.25
    subject_scene_max_boxes: int = 8
    subject_scene_min_mask_area_ratio: float = 0.001
    subject_scene_max_mask_area_ratio: float = 0.95

    subject_mask_frames: int = 32
    subject_mask_min_valid_frames: int = 3
    subject_mask_min_video_success_rate: float = 0.30
    subject_mask_dilation_px: int = 5
    subject_mask_erosion_px: int = 2

    cotracker_checkpoint: str = "models/scaled_offline.pth"
    raft_weights: str = "models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    camera_resize_width: int = 416
    camera_resize_height: int = 240
    camera_ransac_reproj_threshold: float = 3.0
    geometry_shadow_camera_algorithm_version: str = "camera_subject_tracks_v3"
    geometry_shadow_motion_algorithm_version: str = "subject_tracks_canonical_flow_v4_1"
    geometry_shadow_anchor_count: int = 4
    geometry_shadow_subject_grid_size: int = 10
    geometry_shadow_background_grid_size: int = 12
    geometry_shadow_min_background_points: int = 20
    geometry_shadow_min_subject_points: int = 12
    geometry_shadow_min_valid_pairs: int = 2
    geometry_shadow_min_valid_pair_fraction: float = 0.25
    geometry_shadow_background_confidence_threshold: float = 0.35
    geometry_shadow_subject_inlier_ratio_threshold: float = 0.50
    geometry_shadow_homography_min_inliers: int = 30
    geometry_shadow_homography_min_inlier_ratio: float = 0.60
    geometry_shadow_homography_improvement: float = 0.25
    geometry_shadow_canonical_size: int = 256
    geometry_shadow_crop_expansion: float = 0.20
    geometry_shadow_canonical_smoothing_window: int = 5
    geometry_shadow_max_deformation_pairs: int = 16
    geometry_shadow_trajectory_activity_threshold: float = 0.02
    geometry_shadow_deformation_residual_threshold: float = 0.003
    motion_subject_erosion_px: int = 6
    motion_subject_affine_ransac_threshold_px: float = 3.0
    motion_subject_affine_max_points: int = 4000
    motion_min_moving_ratio: float = 0.02
    motion_activity_full_ratio: float = 0.10
    geometry_shadow_camera_scales: tuple[float, ...] = (
        0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03,
        0.03, 0.03, 0.03, 0.05, 0.02, 0.02, 0.02, 0.03,
    )
    geometry_shadow_trajectory_scales: tuple[float, ...] = (
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.50, 0.25,
        0.25, 0.25, 0.25, 0.10, 0.10, 0.10, 0.20, 0.20,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    )
    geometry_shadow_deformation_scales: tuple[float, ...] = (
        0.02, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
    )
    geometry_shadow_camera_kernel_sigma: float = 2.0
    geometry_shadow_trajectory_kernel_sigma: float = 3.0
    geometry_shadow_deformation_kernel_sigma: float = 3.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = EvalConfig()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
