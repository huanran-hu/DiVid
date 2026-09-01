"""Lazy model loading for DiVid's six evaluators."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, EvalConfig, resolve_project_path


@dataclass
class SemanticBundle:
    model: Any
    preprocess: Any
    tokenizer: Any


@dataclass
class StyleBundle:
    model: Any
    transform: Any


@dataclass
class FlowBundle:
    model: Any
    transform: Any


@dataclass
class SubjectSceneBundle:
    grounding_processor: Any
    grounding_model: Any
    sam_processor: Any
    sam_model: Any
    dinov2_processor: Any
    dinov2_model: Any


@dataclass
class TrackerBundle:
    model: Any


class ModelBundle:
    """Load each model only when its dimension is requested."""

    def __init__(self, *, device: str = "cuda", config: EvalConfig = DEFAULT_CONFIG) -> None:
        self.device = device
        self.config = config
        self._semantic = None
        self._style = None
        self._raft = None
        self._tracker = None
        self._subject_scene = None

    def __enter__(self) -> "ModelBundle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._semantic = None
        self._style = None
        self._raft = None
        self._tracker = None
        self._subject_scene = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def semantic(self) -> SemanticBundle:
        if self._semantic is None:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model,
                pretrained=self.config.clip_pretrained,
            )
            tokenizer = open_clip.get_tokenizer(self.config.clip_model)
            self._semantic = SemanticBundle(model.to(self.device).eval(), preprocess, tokenizer)
        return self._semantic

    @property
    def style(self) -> StyleBundle:
        if self._style is None:
            import torch

            from .style import InceptionPool3, inception_transform

            checkpoint = resolve_project_path(self.config.inception_weights)
            model = InceptionPool3()
            if checkpoint.is_file():
                model.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            else:
                from torchvision.models import Inception_V3_Weights, inception_v3

                model.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            self._style = StyleBundle(model.to(self.device).eval(), inception_transform())
        return self._style

    @property
    def raft(self) -> FlowBundle:
        if self._raft is None:
            import torch
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            checkpoint = resolve_project_path(self.config.raft_weights)
            weights = Raft_Large_Weights.C_T_SKHT_V2
            model = raft_large(weights=None, progress=False)
            if checkpoint.is_file():
                model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            else:
                model = raft_large(weights=weights, progress=True)
            self._raft = FlowBundle(model.to(self.device).eval(), weights.transforms())
        return self._raft

    @property
    def tracker(self) -> TrackerBundle:
        if self._tracker is None:
            import sys
            import torch

            checkpoint = resolve_project_path(self.config.cotracker_checkpoint)
            try:
                from cotracker.predictor import CoTrackerPredictor

                model = CoTrackerPredictor(checkpoint=str(checkpoint) if checkpoint.is_file() else None)
            except ImportError as exc:
                raise ImportError(
                    "Install CoTracker3 with: pip install git+https://github.com/facebookresearch/co-tracker.git"
                ) from exc
            if checkpoint.is_file() and not model.__class__.__name__.lower().endswith("predictor"):
                state = torch.load(checkpoint, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
            self._tracker = TrackerBundle(model.to(self.device).eval())
        return self._tracker

    @property
    def subject_scene(self) -> SubjectSceneBundle:
        if self._subject_scene is None:
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Dinov2Model, GroundingDinoForObjectDetection

            grounding_processor = AutoProcessor.from_pretrained(self.config.subject_scene_grounding_model)
            grounding_model = GroundingDinoForObjectDetection.from_pretrained(
                self.config.subject_scene_grounding_model
            ).to(self.device).eval()
            sam_processor = AutoProcessor.from_pretrained(self.config.subject_scene_sam2_model)
            sam_model = AutoModel.from_pretrained(self.config.subject_scene_sam2_model).to(self.device).eval()
            dinov2_processor = AutoImageProcessor.from_pretrained(self.config.subject_scene_dinov2_model)
            dinov2_model = Dinov2Model.from_pretrained(self.config.subject_scene_dinov2_model).to(self.device).eval()
            self._subject_scene = SubjectSceneBundle(
                grounding_processor,
                grounding_model,
                sam_processor,
                sam_model,
                dinov2_processor,
                dinov2_model,
            )
        return self._subject_scene
