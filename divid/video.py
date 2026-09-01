"""Video and frame-directory input helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_videos(directory: str | Path) -> list[Path]:
    path = Path(directory)
    videos = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in {path}.")
    return videos


def frame_directory(video_path: str | Path) -> Path:
    path = Path(video_path)
    return path if path.is_dir() else path.parent / path.stem


def list_frames(directory: str | Path) -> list[Path]:
    path = Path(directory)
    frames = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=lambda item: (int(item.stem) if item.stem.isdigit() else item.name))
    if not frames:
        raise FileNotFoundError(f"No frame images found in {path}.")
    return frames


def video_metadata(source: str | Path) -> dict[str, float | int | str]:
    path = Path(source)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video {path}.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps > 0 else 0.0,
    }


def _frame_indices(frame_count: int, count: int, minimum: int = 1) -> list[int]:
    if frame_count <= 0:
        raise ValueError("The input contains no frames.")
    target = min(max(int(count), int(minimum)), frame_count)
    return np.linspace(0, frame_count - 1, target).round().astype(int).tolist()


def _read_rgb(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read frame {path}.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    return rgb


def sample_frames(
    source: str | Path,
    count: int,
    *,
    minimum: int = 1,
    target_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Uniformly sample RGB frames from a video or its sibling frame directory."""

    path = Path(source)
    frame_dir = frame_directory(path)
    if frame_dir.is_dir():
        paths = list_frames(frame_dir)
        return [_read_rgb(paths[index], target_size) for index in _frame_indices(len(paths), count, minimum)]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video {path}.")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Could not determine frame count for {path}.")

    frames: list[np.ndarray] = []
    last: np.ndarray | None = None
    for index in _frame_indices(frame_count, count, minimum):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if not ok:
            if last is None:
                continue
            rgb = last.copy()
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last = rgb
        if target_size is not None:
            rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise ValueError(f"Could not read frames from {path}.")
    while len(frames) < minimum:
        frames.append(frames[-1].copy())
    return frames


def read_frame_images(source: str | Path, count: int, minimum: int = 1) -> list:
    """Read PIL frames for the segmentation pipeline."""

    from PIL import Image

    return [Image.fromarray(frame) for frame in sample_frames(source, count, minimum=minimum)]
