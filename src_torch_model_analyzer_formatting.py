from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class ColorScheme(Enum):
    """Color schemes for formatting the model summary output."""
    
    DEFAULT = {
        'header': '\033[95m',
        'layer': '\033[94m',
        'stats': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'end': '\033[0m'
    }
    
    MONOCHROME = {
        'header': '',
        'layer': '',
        'stats': '',
        'warning': '',
        'error': '',
        'end': ''
    }

@dataclass
class FormattingOptions:
    """Configuration for summary formatting."""
    
    show_layer_indices: bool = True
    show_param_count: bool = True
    show_trainable: bool = True
    show_shapes: bool = True
    show_flops: bool = True
    show_memory: bool = True
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    max_depth: int = None
    compact_mode: bool = False
    unicode_icons: bool = True
    
    def get_color(self, color_type: str) -> str:
        """Get ANSI color code for given type."""
        return self.color_scheme.value.get(color_type, '')