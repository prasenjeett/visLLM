"""
Torch Model Analyzer
-------------------

A comprehensive PyTorch model analysis tool.
"""

from .core import model_summary, ModelSummary
from .formatting import FormattingOptions, ColorScheme

__version__ = "0.1.0"
__author__ = "Prasenjeet"
__email__ = "prasenjeett@example.com"

__all__ = [
    "model_summary",
    "ModelSummary",
    "FormattingOptions",
    "ColorScheme",
]
