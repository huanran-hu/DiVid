"""DiVid: six-dimensional video diversity evaluation."""

from .config import DEFAULT_CONFIG, EvalConfig
from .evaluate import DIMENSIONS, evaluate

__all__ = ["DEFAULT_CONFIG", "DIMENSIONS", "EvalConfig", "evaluate"]

__version__ = "0.1.0"
