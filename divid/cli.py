"""Command-line interface for DiVid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG
from .evaluate import DIMENSIONS, evaluate
from .video import list_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate six-dimensional video diversity.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-dir", type=Path, help="Directory containing the generated videos.")
    group.add_argument("--videos", type=Path, nargs="+", help="Explicit video paths.")
    parser.add_argument("--prompt", required=True, help="The prompt used to generate this video set.")
    parser.add_argument("--subjects", nargs="*", help="Foreground queries for Subject/Scene/Motion/Camera.")
    parser.add_argument("--dimensions", nargs="+", choices=DIMENSIONS, default=list(DIMENSIONS))
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    videos = args.videos if args.videos is not None else list_videos(args.video_dir)
    result = evaluate(
        videos,
        args.prompt,
        subjects=args.subjects,
        dimensions=args.dimensions,
        device=args.device,
        config=DEFAULT_CONFIG,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
