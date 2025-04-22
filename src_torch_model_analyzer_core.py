import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from typing import Tuple, List, Dict, Union, Any, Optional
from .utils import (calculate_conv_flops, calculate_linear_flops,
                   estimate_memory_usage, format_memory, generate_graphviz_dot)
from .formatting import FormattingOptions, ColorScheme
import json

class ModelSummary:
    """
    Enhanced PyTorch model summary with advanced features.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 input_size: Tuple, 
                 batch_size: int = 1,
                 formatting_options: Optional[FormattingOptions] = None,
                 device: str = 'cpu'):
        """
        Initialize the enhanced ModelSummary class.
        
        Args:
            model (nn.Module): PyTorch model to summarize
            input_size (tuple): Input size excluding batch size (C, H, W) or (C, L)
            batch_size (int): Batch size for input
            formatting_options (FormattingOptions): Custom formatting options
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.formatting = formatting_options or FormattingOptions()
        
        self.summary = OrderedDict()
        self.hooks = []
        self.total_params = 0
        self.trainable_params = 0
        self.total_flops = 0
        self.total_memory = 0
        self.layer_memory = {}
        self.dependencies = []
        
    def _register_hooks(self):
        """Register hooks for all layers."""
        def pre_hook(module, input):
            if not input:
                return None
            self.layer_memory[input[0].device] = (
                self.layer_memory.get(input[0].device, 0) +
                estimate_memory_usage(input[0].shape)
            )
            
        def hook(module, input, output):
            if not input or not output:
                return None
                
            class_name = str(module.__class__.__name__)
            layer_idx = len(self.summary)
            
            # Calculate memory usage
            input_memory = estimate_memory_usage(input[0].shape)
            output_memory = estimate_memory_usage(output.shape)
            
            # Calculate FLOPS
            flops = 0
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                flops = calculate_conv_flops(
                    input[0].shape, 
                    output.shape,
                    module.kernel_size,
                    module.groups
                )
            elif isinstance(module, nn.Linear):
                flops = calculate_linear_flops(input[0].shape, output.shape)
            
            # Get layer information
            layer_info = {
                "idx": layer_idx,
                "class_name": class_name,
                "input_shape": self._format_shape(input[0].shape),
                "output_shape": self._format_shape(output.shape),
                "num_params": self._count_params(module),
                "trainable": any(p.requires_grad for p in module.parameters()),
                "flops": flops,
                "input_memory": input_memory,
                "output_memory": output_memory,
            }
            
            self.summary[layer_idx] = layer_info
            self.total_flops += flops
            
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LSTM, nn.RNN, nn.GRU)):
                self.hooks.append(module.register_forward_pre_hook(pre_hook))
                self.hooks.append(module.register_forward_hook(hook))

    def _format_shape(self, shape: Tuple) -> str:
        """Format tensor shape into a string."""
        return str(tuple(shape))
    
    def _count_params(self, module: nn.Module) -> int:
        """Count number of parameters in a module."""
        return sum(p.numel() for p in module.parameters())
    
    def generate_summary(self) -> None:
        """Generate the model summary with all features."""
        self._register_hooks()
        
        # Create dummy input
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            self.model(x)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        # Calculate totals
        for layer in self.summary.values():
            self.total_params += layer["num_params"]
            if layer["trainable"]:
                self.trainable_params += layer["num_params"]
        
        self.total_memory = sum(self.layer_memory.values())
    
    def export_json(self, filepath: str) -> None:
        """Export summary to JSON file."""
        data = {
            "model_summary": self.summary,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_flops": self.total_flops,
            "total_memory": self.total_memory,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_graph(self, filepath: str) -> None:
        """Export model architecture graph to DOT file."""
        dot_str = generate_graphviz_dot(list(self.summary.values()))
        with open(filepath, 'w') as f:
            f.write(dot_str)
    
    def __str__(self) -> str:
        """Generate formatted summary string."""
        fmt = self.formatting
        c = fmt.color_scheme.value
        
        header = f"\n{c['header']}Model Summary{c['end']}\n{'-' * 120}\n"
        
        # Define columns based on formatting options
        columns = []
        if fmt.show_layer_indices:
            columns.append(("Idx", 5))
        columns.extend([
            ("Layer (type)", 35),
            ("Output Shape", 25) if fmt.show_shapes else None,
            ("Param #", 12) if fmt.show_param_count else None,
            ("FLOPS", 15) if fmt.show_flops else None,
            ("Memory (MB)", 12) if fmt.show_memory else None,
            ("Trainable", 10) if fmt.show_trainable else None,
        ])
        columns = [col for col in columns if col is not None]
        
        # Create row format string
        row_format = "".join("{:<" + str(width) + "} " for _, width in columns)
        
        # Build header
        summary_str = header
        summary_str += row_format.format(*(name for name, _ in columns)) + "\n"
        summary_str += "=" * 120 + "\n"
        
        # Add layer information
        for layer in self.summary.values():
            values = []
            for col_name, _ in columns:
                if col_name == "Idx":
                    values.append(str(layer["idx"]))
                elif col_name == "Layer (type)":
                    values.append(f"{layer['class_name']}")
                elif col_name == "Output Shape":
                    values.append(layer["output_shape"])
                elif col_name == "Param #":
                    values.append(f"{layer['num_params']:,}")
                elif col_name == "FLOPS":
                    values.append(f"{layer['flops']:,}")
                elif col_name == "Memory (MB)":
                    values.append(f"{layer['output_memory']:.2f}")
                elif col_name == "Trainable":
                    values.append(str(layer["trainable"]))
            
            summary_str += row_format.format(*values) + "\n"
        
        # Add summary statistics
        summary_str += "-" * 120 + "\n"
        stats = [
            (f"{c['stats']}Total params:{c['end']}", f"{self.total_params:,}"),
            (f"{c['stats']}Trainable params:{c['end']}", f"{self.trainable_params:,}"),
            (f"{c['stats']}Non-trainable params:{c['end']}", 
             f"{self.total_params - self.trainable_params:,}"),
            (f"{c['stats']}Total FLOPS:{c['end']}", f"{self.total_flops:,}"),
            (f"{c['stats']}Total Memory:{c['end']}", format_memory(self.total_memory))
        ]
        
        max_label_len = max(len(label) for label, _ in stats)
        for label, value in stats:
            summary_str += f"{label:<{max_label_len + 10}} {value}\n"
        
        return summary_str

def model_summary(model: nn.Module, 
                 input_size: Tuple, 
                 batch_size: int = 1,
                 formatting_options: Optional[FormattingOptions] = None,
                 device: str = 'cpu',
                 export_json: str = None,
                 export_graph: str = None) -> str:
    """
    Enhanced convenience function to get model summary.
    
    Args:
        model (nn.Module): PyTorch model to summarize
        input_size (tuple): Input size excluding batch size (C, H, W) or (C, L)
        batch_size (int): Batch size for input
        formatting_options (FormattingOptions): Custom formatting options
        device (str): Device to run the model on
        export_json (str): Optional filepath to export JSON summary
        export_graph (str): Optional filepath to export architecture graph
        
    Returns:
        str: Formatted model summary
    """
    summary = ModelSummary(model, input_size, batch_size, formatting_options, device)
    summary.generate_summary()
    
    if export_json:
        summary.export_json(export_json)
    if export_graph:
        summary.export_graph(export_graph)
        
    return str(summary)