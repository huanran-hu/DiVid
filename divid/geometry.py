"""Latest Motion v4.1 and Camera v3 implementation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_CONFIG, EvalConfig
from .masks import SubjectMaskTracks, morph_mask, resize_mask
from .math_utils import rbf_metrics, upper_triangle_mean, vendi_score_from_kernel
from .motion_core import affine_flow, direction_histogram, motion_activity_weight, raft_flow


def _homography(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return np.vstack([values, [0.0, 0.0, 1.0]]) if values.shape == (2, 3) else values


def transform_points(matrix: np.ndarray, points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ _homography(matrix).T
    denominator = projected[:, 2:3]
    denominator = np.where(np.abs(denominator) > eps, denominator, eps)
    return projected[:, :2] / denominator


def _transfer_error(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
    if len(source) == 0:
        return math.inf
    forward = np.linalg.norm(transform_points(matrix, source) - target, axis=1)
    try:
        inverse = np.linalg.inv(_homography(matrix))
    except np.linalg.LinAlgError:
        return math.inf
    backward = np.linalg.norm(transform_points(inverse, target) - source, axis=1)
    return float(np.mean(0.5 * (forward + backward)))


def _fit_background(source: np.ndarray, target: np.ndarray, config: EvalConfig) -> dict:
    minimum = int(config.geometry_shadow_min_background_points)
    if len(source) < minimum:
        return {"valid": False, "status": "not_enough_background_points"}
    affine, affine_inliers = cv2.estimateAffinePartial2D(
        source.astype(np.float32),
        target.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=config.camera_ransac_reproj_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or affine_inliers is None:
        return {"valid": False, "status": "affine_failed"}
    affine_h = _homography(affine)
    affine_mask = affine_inliers.ravel().astype(bool)
    affine_error = _transfer_error(affine_h, source[affine_mask], target[affine_mask])

    homography, homography_inliers = cv2.findHomography(
        source.astype(np.float32),
        target.astype(np.float32),
        cv2.RANSAC,
        config.camera_ransac_reproj_threshold,
        maxIters=2000,
        confidence=0.995,
    )
    homography_mask = (
        homography_inliers.ravel().astype(bool)
        if homography is not None and homography_inliers is not None
        else np.zeros(len(source), dtype=bool)
    )
    homography_error = (
        _transfer_error(homography, source[homography_mask], target[homography_mask])
        if homography is not None and homography_mask.sum() >= 4
        else math.inf
    )
    use_homography = (
        homography is not None
        and homography_mask.sum() >= config.geometry_shadow_homography_min_inliers
        and homography_mask.mean() >= config.geometry_shadow_homography_min_inlier_ratio
        and homography_error <= affine_error * (1.0 - config.geometry_shadow_homography_improvement)
    )
    matrix = np.asarray(homography if use_homography else affine_h, dtype=np.float64)
    inliers = homography_mask if use_homography else affine_mask
    residual = homography_error if use_homography else affine_error
    confidence = float(np.clip(inliers.mean() / (1.0 + max(residual, 0.0) / 3.0), 0.0, 1.0))
    return {
        "valid": int(inliers.sum()) >= minimum,
        "status": "ok",
        "matrix": matrix,
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "residual": float(residual),
        "confidence": confidence,
    }


def _motion_parameters(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    center = np.asarray([[0.5 * width, 0.5 * height]])
    projected = transform_points(matrix, center)[0]
    basis_x = transform_points(matrix, center + [[1.0, 0.0]])[0] - projected
    basis_y = transform_points(matrix, center + [[0.0, 1.0]])[0] - projected
    jacobian = np.column_stack([basis_x, basis_y])
    scale = math.sqrt(max(abs(float(np.linalg.det(jacobian))), 1e-12))
    rotation = math.atan2(float(jacobian[1, 0]), float(jacobian[0, 0]))
    displacement = projected - center[0]
    return np.asarray([displacement[0] / width, displacement[1] / height, math.log(scale), rotation])


def _camera_descriptor(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(16, dtype=np.float64)
    values = np.asarray(vectors, dtype=np.float64)
    return np.concatenate([values.mean(0), values.std(0), np.abs(values).mean(0), np.percentile(np.abs(values), 90, axis=0)])


def _anchors(frame_count: int, count: int) -> list[int]:
    count = min(frame_count, max(1, count))
    return sorted({min(frame_count - 1, index * frame_count // count) for index in range(count)})


def _grid_queries(mask: np.ndarray, grid_size: int, frame_index: int, minimum: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    target = max(minimum, grid_size * grid_size)
    grid_x, grid_y = np.meshgrid(np.linspace(xs.min(), xs.max(), grid_size), np.linspace(ys.min(), ys.max(), grid_size))
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    ix = np.clip(np.round(candidates[:, 0]).astype(int), 0, mask.shape[1] - 1)
    iy = np.clip(np.round(candidates[:, 1]).astype(int), 0, mask.shape[0] - 1)
    selected = {(int(round(x)), int(round(y))) for x, y in candidates[mask[iy, ix]]}
    if len(selected) < target:
        pixels = np.stack([xs, ys], axis=1)
        for x, y in pixels[np.linspace(0, len(pixels) - 1, min(len(pixels), target * 4)).round().astype(int)]:
            selected.add((int(x), int(y)))
            if len(selected) >= target:
                break
    points = np.asarray(sorted(selected), dtype=np.float32)
    if len(points) > target:
        points = points[np.linspace(0, len(points) - 1, target).round().astype(int)]
    return np.concatenate([np.full((len(points), 1), frame_index, dtype=np.float32), points], axis=1)


def _build_queries(masks: list[np.ndarray], config: EvalConfig) -> tuple[np.ndarray, np.ndarray]:
    subject, background = [], []
    for index in _anchors(len(masks), config.geometry_shadow_anchor_count):
        foreground = morph_mask(masks[index], config.motion_subject_erosion_px, operation="erode")
        if not foreground.any():
            foreground = np.asarray(masks[index], dtype=bool)
        excluded = morph_mask(masks[index], config.subject_mask_dilation_px, operation="dilate")
        subject.append(_grid_queries(foreground, config.geometry_shadow_subject_grid_size, index, config.geometry_shadow_min_subject_points))
        background.append(_grid_queries(~excluded, config.geometry_shadow_background_grid_size, index, config.geometry_shadow_min_background_points))
    return np.concatenate(subject), np.concatenate(background)


def _fit_subject(source: np.ndarray, target: np.ndarray, minimum: int) -> dict:
    if len(source) < minimum:
        return {"valid": False}
    matrix, inliers = cv2.estimateAffinePartial2D(
        source.astype(np.float32),
        target.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return {"valid": False}
    mask = inliers.ravel().astype(bool)
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    return {
        "valid": int(mask.sum()) >= minimum,
        "inlier_ratio": float(mask.mean()),
        "log_scale": math.log(max(math.sqrt(a * a + b * b), 1e-8)),
        "rotation": math.atan2(float(matrix[1, 0]), a),
    }


def _bbox_diagonal(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    return float(max(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), 1.0)) if len(xs) else 1.0


def _trajectory_descriptor(records: list[dict], config: EvalConfig) -> np.ndarray:
    if not records:
        return np.zeros(24, dtype=np.float64)
    velocity = np.asarray([[item["vx"], item["vy"]] for item in records])
    speed = np.linalg.norm(velocity, axis=1)
    dt = np.asarray([item["dt"] for item in records])
    displacement = velocity * dt[:, None]
    net = displacement.sum(axis=0)
    path = float(np.linalg.norm(displacement, axis=1).sum())
    direction = np.arctan2(velocity[:, 1], velocity[:, 0])
    turns = [abs(math.atan2(math.sin(b - a), math.cos(b - a))) for a, b in zip(direction[:-1], direction[1:])]
    scale = np.asarray([item["log_scale"] for item in records])
    rotation = np.asarray([item["rotation"] for item in records])
    prefix = np.asarray([
        velocity[:, 0].mean(), velocity[:, 1].mean(), velocity[:, 0].std(), velocity[:, 1].std(),
        speed.mean(), speed.std(), np.percentile(speed, 90),
        np.mean(speed > config.geometry_shadow_trajectory_activity_threshold),
        np.linalg.norm(net) / max(dt.sum(), config.eps), np.linalg.norm(net) / max(path, config.eps),
        np.mean(turns) if turns else 0.0, scale.mean(), scale.std(), np.abs(scale).mean(),
        rotation.mean(), rotation.std(),
    ])
    return np.concatenate([prefix, direction_histogram(direction, np.maximum(speed, config.eps), 8)])


def _required_pairs(total: int, config: EvalConfig) -> int:
    return max(config.geometry_shadow_min_valid_pairs, math.ceil(total * config.geometry_shadow_min_valid_pair_fraction))


def _track_geometry(track, tracker, device: str, config: EvalConfig) -> dict:
    size = (config.camera_resize_width, config.camera_resize_height)
    frames = [np.asarray(frame.resize(size).convert("RGB"), dtype=np.uint8) for frame in track.frames]
    masks = [resize_mask(mask, size) for mask in track.union_masks]
    subject_queries, background_queries = _build_queries(masks, config)
    if len(subject_queries) < config.geometry_shadow_min_subject_points or len(background_queries) < config.geometry_shadow_min_background_points:
        return {"camera_valid": False, "trajectory_valid": False}
    queries = np.concatenate([subject_queries, background_queries])
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    query_tensor = torch.from_numpy(queries).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        tracks_tensor, visible_tensor = tracker(video, queries=query_tensor, backward_tracking=True)
    points = tracks_tensor[0].float().cpu().numpy()
    visible = visible_tensor[0].cpu().numpy().astype(bool)
    subject_count = len(subject_queries)
    camera_vectors, trajectory_records = [], []
    dt = 1.0 / max(config.frame_dir_fps, config.eps)
    for index in range(len(frames) - 1):
        bg_eligible = background_queries[:, 0] <= index
        bg_valid = bg_eligible & visible[index, subject_count:] & visible[index + 1, subject_count:]
        background = _fit_background(points[index, subject_count:][bg_valid], points[index + 1, subject_count:][bg_valid], config)
        if not background.get("valid") or background["confidence"] < config.geometry_shadow_background_confidence_threshold:
            continue
        camera_vectors.append(_motion_parameters(background["matrix"], *size))
        subject_eligible = subject_queries[:, 0] <= index
        subject_valid = subject_eligible & visible[index, :subject_count] & visible[index + 1, :subject_count]
        source = points[index, :subject_count][subject_valid]
        target = points[index + 1, :subject_count][subject_valid]
        stabilized = transform_points(background["matrix"], source)
        fitted = _fit_subject(stabilized, target, config.geometry_shadow_min_subject_points)
        if not fitted.get("valid") or fitted["inlier_ratio"] < config.geometry_shadow_subject_inlier_ratio_threshold:
            continue
        translation = np.median(target - stabilized, axis=0)
        velocity = translation / max(_bbox_diagonal(masks[index]) * dt, config.eps)
        trajectory_records.append({
            "vx": float(velocity[0]), "vy": float(velocity[1]), "dt": dt,
            "log_scale": fitted["log_scale"], "rotation": fitted["rotation"],
        })
    return {
        "camera_valid": len(camera_vectors) >= _required_pairs(len(frames) - 1, config),
        "trajectory_valid": len(trajectory_records) >= _required_pairs(len(frames) - 1, config),
        "camera_vector": _camera_descriptor(camera_vectors),
        "trajectory_vector": _trajectory_descriptor(trajectory_records, config),
    }


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    radius = max(window // 2, 0)
    return np.asarray([np.median(values[max(0, i - radius): i + radius + 1]) for i in range(len(values))])


def _canonical_frames(track, config: EvalConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes = []
    for mask in track.union_masks:
        ys, xs = np.nonzero(mask)
        boxes.append((0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()), np.ptp(xs) + 1, np.ptp(ys) + 1))
    values = np.asarray(boxes, dtype=np.float64)
    values = np.stack([_smooth(values[:, channel], config.geometry_shadow_canonical_smoothing_window) for channel in range(4)], axis=1)
    result = []
    size = config.geometry_shadow_canonical_size
    for frame, mask, (center_x, center_y, width, height) in zip(track.frames, track.union_masks, values):
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        mask = resize_mask(mask, frame.size)
        expansion = config.geometry_shadow_crop_expansion
        scale = min(size / max(width * (1 + 2 * expansion), 1), size / max(height * (1 + 2 * expansion), 1))
        matrix = np.asarray([[scale, 0, 0.5 * size - scale * center_x], [0, scale, 0.5 * size - scale * center_y]], dtype=np.float32)
        fill = rgb[mask].mean(axis=0) if mask.any() else rgb.mean(axis=(0, 1))
        masked = rgb.copy()
        masked[~mask] = fill.astype(np.uint8)
        image = cv2.warpAffine(masked, matrix, (size, size), borderValue=tuple(float(x) for x in fill))
        canvas_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (size, size), flags=cv2.INTER_NEAREST) > 0
        result.append((image, canvas_mask))
    return result


def _deformation_pair(flow: np.ndarray, mask: np.ndarray, config: EvalConfig) -> tuple[np.ndarray, bool]:
    subject = morph_mask(mask, config.subject_mask_erosion_px, operation="erode")
    ys, xs = np.nonzero(subject)
    if len(xs) < config.geometry_shadow_min_subject_points:
        return np.zeros(10), False
    count = min(len(xs), config.motion_subject_affine_max_points)
    indices = np.linspace(0, len(xs) - 1, count).round().astype(int)
    source = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
    target = source + flow[:, ys[indices], xs[indices]].T
    matrix, _ = cv2.estimateAffinePartial2D(source, target, cv2.RANSAC, config.motion_subject_affine_ransac_threshold_px)
    residual = flow - (affine_flow(matrix, flow.shape[2], flow.shape[1]) if matrix is not None else 0)
    u, v = residual[0, subject], residual[1, subject]
    magnitude = np.sqrt(u * u + v * v) / max(np.hypot(flow.shape[2], flow.shape[1]), config.eps)
    excess = np.clip(magnitude - config.geometry_shadow_deformation_residual_threshold, 0, None)
    moving = excess > 0
    ratio = float(moving.mean()) if moving.size else 0.0
    activity = motion_activity_weight(ratio, config.motion_min_moving_ratio, config.motion_activity_full_ratio, config.eps)
    histogram = direction_histogram(np.arctan2(v[moving], u[moving]), excess[moving], 8) * activity if moving.any() else np.zeros(8)
    mean = float(excess[moving].mean()) if moving.any() else 0.0
    return np.concatenate([[mean, ratio], histogram]), True


def _deformation_descriptor(track, raft_model, raft_transform, device: str, config: EvalConfig) -> tuple[np.ndarray, bool]:
    canonical = _canonical_frames(track, config)
    pair_count = min(config.geometry_shadow_max_deformation_pairs, len(canonical) - 1)
    indices = sorted(set(np.linspace(0, len(canonical) - 2, pair_count).round().astype(int).tolist()))
    features = []
    for index in indices:
        flow = raft_flow(raft_model, raft_transform, canonical[index][0], canonical[index + 1][0], device=device)
        feature, valid = _deformation_pair(flow, canonical[index][1], config)
        if valid:
            features.append(feature)
    if len(features) < _required_pairs(len(indices), config):
        return np.zeros(20), False
    values = np.asarray(features)
    return np.concatenate([values[:, :2].mean(0), values[:, :2].std(0), values[:, 2:].mean(0), values[:, 2:].std(0)]), True


def compute_motion_camera(
    video_paths: list[str | Path],
    *,
    device: str,
    config: EvalConfig = DEFAULT_CONFIG,
    bundle=None,
    mask_tracks: SubjectMaskTracks,
) -> dict:
    if mask_tracks.status != "ok":
        raise ValueError("Valid subject masks are required for Motion and Camera diversity.")
    if bundle is None:
        from .models import ModelBundle

        with ModelBundle(device=device, config=config) as owned:
            return compute_motion_camera(video_paths, device=device, config=config, bundle=owned, mask_tracks=mask_tracks)

    camera_vectors, camera_indices = [], []
    trajectory_vectors, deformation_vectors, motion_indices = [], [], []
    tracker = bundle.tracker.model
    raft = bundle.raft
    for index, track in zip(mask_tracks.valid_indices, mask_tracks.valid_tracks):
        geometry = _track_geometry(track, tracker, device, config)
        if geometry["camera_valid"]:
            camera_indices.append(index)
            camera_vectors.append(geometry["camera_vector"])
        deformation, deformation_valid = _deformation_descriptor(track, raft.model, raft.transform, device, config)
        if geometry["trajectory_valid"] and deformation_valid:
            motion_indices.append(index)
            trajectory_vectors.append(geometry["trajectory_vector"])
            deformation_vectors.append(deformation)

    if len(camera_vectors) < 2:
        raise ValueError("Fewer than two videos passed Camera v3 geometry checks.")
    if len(trajectory_vectors) < 2:
        raise ValueError("Fewer than two videos passed Motion v4.1 geometry checks.")

    camera = rbf_metrics(np.asarray(camera_vectors), config.geometry_shadow_camera_scales, config.geometry_shadow_camera_kernel_sigma, config.eps)
    trajectory = rbf_metrics(np.asarray(trajectory_vectors), config.geometry_shadow_trajectory_scales, config.geometry_shadow_trajectory_kernel_sigma, config.eps)
    deformation = rbf_metrics(np.asarray(deformation_vectors), config.geometry_shadow_deformation_scales, config.geometry_shadow_deformation_kernel_sigma, config.eps)
    motion_similarity = 0.5 * trajectory["similarity"] + 0.5 * deformation["similarity"]
    np.fill_diagonal(motion_similarity, 1.0)
    motion_distance = 1.0 - motion_similarity
    np.fill_diagonal(motion_distance, 0.0)
    return {
        "camera_mpd": camera["mpd"],
        "camera_vendi": camera["vendi"],
        "camera_similarity": camera["similarity"].tolist(),
        "camera_distance": camera["distance"].tolist(),
        "camera_valid_videos": len(camera_indices),
        "camera_valid_video_indices": camera_indices,
        "camera_algorithm": config.geometry_shadow_camera_algorithm_version,
        "motion_mpd": upper_triangle_mean(motion_distance),
        "motion_vendi": vendi_score_from_kernel(motion_similarity, eps=config.eps),
        "motion_similarity": motion_similarity.tolist(),
        "motion_distance": motion_distance.tolist(),
        "motion_valid_videos": len(motion_indices),
        "motion_valid_video_indices": motion_indices,
        "motion_algorithm": config.geometry_shadow_motion_algorithm_version,
    }
