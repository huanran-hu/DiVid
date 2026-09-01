"""Prompt-grounded subject masks shared by Subject, Scene, Motion, and Camera."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_CONFIG, EvalConfig
from .video import frame_directory, list_frames, sample_frames


@dataclass
class SubjectVideoTrack:
    video_path: Path
    frames: list[Image.Image]
    frame_names: list[str]
    subjects: tuple[str, ...]
    subject_masks: dict[str, list[np.ndarray]]
    union_masks: list[np.ndarray]
    valid: bool
    error: str | None = None


@dataclass
class SubjectMaskTracks:
    subjects: tuple[str, ...]
    video_tracks: list[SubjectVideoTrack]
    valid_indices: list[int]
    status: str
    error: str | None = None

    @property
    def valid_tracks(self) -> list[SubjectVideoTrack]:
        return [self.video_tracks[index] for index in self.valid_indices]


def normalize_query(value: str) -> str:
    query = str(value).strip().lower()
    return query if query.endswith(".") else f"{query}."


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if np.asarray(mask).shape == (height, width):
        return np.asarray(mask).astype(bool)
    image = Image.fromarray(np.asarray(mask).astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def morph_mask(mask: np.ndarray, pixels: int, *, operation: str) -> np.ndarray:
    if pixels <= 0:
        return np.asarray(mask).astype(bool)
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
    source = np.asarray(mask).astype(np.uint8)
    if operation == "dilate":
        return cv2.dilate(source, kernel, iterations=1).astype(bool)
    if operation == "erode":
        return cv2.erode(source, kernel, iterations=1).astype(bool)
    raise ValueError(f"Unknown mask operation: {operation}")


def is_valid_mask(mask: np.ndarray, config: EvalConfig) -> bool:
    ratio = float(np.asarray(mask).mean()) if np.asarray(mask).size else 0.0
    return config.subject_scene_min_mask_area_ratio <= ratio <= config.subject_scene_max_mask_area_ratio


def _device(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def _detect(image: Image.Image, query: str, bundle: Any, config: EvalConfig, device: str) -> np.ndarray:
    inputs = bundle.grounding_processor(images=image, text=query, return_tensors="pt")
    inputs = _device(inputs, device)
    with torch.inference_mode():
        outputs = bundle.grounding_model(**inputs)
    result = bundle.grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(config.subject_scene_grounding_box_threshold),
        text_threshold=float(config.subject_scene_grounding_text_threshold),
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = result.get("boxes")
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scores = result.get("scores")
    if scores is not None and hasattr(scores, "detach"):
        order = scores.detach().cpu().numpy().argsort()[::-1]
        order = order[: int(config.subject_scene_max_boxes)]
        boxes = boxes.detach().cpu().numpy()[order]
    else:
        boxes = boxes.detach().cpu().numpy()[: int(config.subject_scene_max_boxes)]
    return boxes.astype(np.float32)


def _track_boxes(frames: list[Image.Image], seed: int, boxes: np.ndarray, bundle: Any, device: str) -> list[np.ndarray]:
    object_ids = list(range(1, len(boxes) + 1))
    session = bundle.sam_processor.init_video_session(
        video=frames,
        inference_device=device,
        inference_state_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=torch.float32,
    )
    bundle.sam_processor.add_inputs_to_inference_session(
        session,
        frame_idx=seed,
        obj_ids=object_ids,
        input_boxes=[boxes.astype(float).tolist()],
        original_size=(frames[seed].height, frames[seed].width),
    )
    outputs: dict[int, Any] = {}
    with torch.inference_mode():
        outputs[seed] = bundle.sam_model(session, frame_idx=seed)
        for index in range(seed + 1, len(frames)):
            outputs[index] = bundle.sam_model(session, frame_idx=index, reverse=False)
        for index in range(seed - 1, -1, -1):
            outputs[index] = bundle.sam_model(session, frame_idx=index, reverse=True)

    masks = []
    for index, frame in enumerate(frames):
        values = outputs[index].pred_masks.detach().float().cpu().numpy().squeeze()
        if values.ndim == 2:
            values = values[None, ...]
        elif values.ndim != 3:
            values = values.reshape((-1, values.shape[-2], values.shape[-1]))
        masks.append(resize_mask(np.any(values > 0.0, axis=0), frame.size))
    return masks


def _sample_mask_frames(path: Path, config: EvalConfig) -> tuple[list[Image.Image], list[str]]:
    directory = frame_directory(path)
    if directory.is_dir():
        paths = list_frames(directory)
        target = min(max(config.subject_mask_frames, config.subject_mask_min_valid_frames), len(paths))
        indices = np.linspace(0, len(paths) - 1, target).round().astype(int)
        return [Image.open(paths[int(index)]).convert("RGB") for index in indices], [paths[int(index)].name for index in indices]
    frames = sample_frames(path, config.subject_mask_frames, minimum=config.subject_mask_min_valid_frames)
    return [Image.fromarray(frame) for frame in frames], [f"sampled_{index:04d}" for index in range(len(frames))]


def _subject_masks(subject: str, frames: list[Image.Image], bundle: Any, config: EvalConfig, device: str) -> tuple[list[np.ndarray], str | None]:
    query = normalize_query(subject)
    seed = None
    boxes = None
    for index, frame in enumerate(frames):
        found = _detect(frame, query, bundle, config, device)
        if len(found):
            seed, boxes = index, found
            break
    if seed is None or boxes is None:
        return [np.zeros((frame.height, frame.width), dtype=bool) for frame in frames], "grounding_dino_no_detection"
    try:
        return _track_boxes(frames, seed, boxes, bundle, device), None
    except Exception as exc:
        return [np.zeros((frame.height, frame.width), dtype=bool) for frame in frames], f"sam2_tracking_failed: {exc!r}"


def compute_subject_mask_tracks(
    video_paths: list[str | Path],
    subjects: tuple[str, ...] | list[str],
    *,
    device: str,
    config: EvalConfig = DEFAULT_CONFIG,
    bundle=None,
) -> SubjectMaskTracks:
    normalized = tuple(item.strip() for item in subjects if str(item).strip())
    if not normalized:
        raise ValueError("At least one subject query is required for the four subject-dependent dimensions.")
    if bundle is None:
        from .models import ModelBundle

        with ModelBundle(device=device, config=config) as owned:
            return compute_subject_mask_tracks(video_paths, normalized, device=device, config=config, bundle=owned)

    model_bundle = bundle.subject_scene
    tracks: list[SubjectVideoTrack] = []
    valid_indices: list[int] = []
    for video_index, source in enumerate(video_paths):
        path = Path(source)
        try:
            frames, frame_names = _sample_mask_frames(path, config)
            masks_by_subject: dict[str, list[np.ndarray]] = {}
            errors: list[str] = []
            for subject in normalized:
                masks, error = _subject_masks(subject, frames, model_bundle, config, device)
                masks_by_subject[subject] = masks
                if error:
                    errors.append(f"{subject}: {error}")
                elif sum(is_valid_mask(mask, config) for mask in masks) < config.subject_mask_min_valid_frames:
                    errors.append(f"{subject}: insufficient valid mask frames")
            union_masks = []
            for index, frame in enumerate(frames):
                union = np.zeros((frame.height, frame.width), dtype=bool)
                for subject in normalized:
                    union |= resize_mask(masks_by_subject[subject][index], frame.size)
                union_masks.append(union)
            track = SubjectVideoTrack(
                path,
                frames,
                frame_names,
                normalized,
                masks_by_subject,
                union_masks,
                not errors,
                "; ".join(errors) if errors else None,
            )
        except Exception as exc:
            frames = []
            track = SubjectVideoTrack(path, frames, [], normalized, {}, [], False, repr(exc))
        tracks.append(track)
        if track.valid:
            valid_indices.append(video_index)

    if len(valid_indices) < 2:
        return SubjectMaskTracks(normalized, tracks, valid_indices, "error", "Fewer than two videos have valid subject masks.")
    return SubjectMaskTracks(normalized, tracks, valid_indices, "ok")
