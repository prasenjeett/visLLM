import math
from typing import List, Tuple, Union, Any
import numpy as np

def calculate_conv_flops(input_shape: Tuple[int, ...], 
                        output_shape: Tuple[int, ...],
                        kernel_size: Union[int, Tuple[int, ...]],
                        groups: int = 1) -> int:
    """Calculate FLOPS for convolutional layers."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
        
    # Extract shapes
    batch_size, in_channels, in_height, in_width = input_shape
    _, out_channels, out_height, out_width = output_shape
    
    # Calculate FLOPS
    flops = (2 * kernel_size[0] * kernel_size[1] * in_channels * out_channels *
             out_height * out_width) // groups
    
    return flops * batch_size

def calculate_linear_flops(input_shape: Tuple[int, ...], 
                         output_shape: Tuple[int, ...]) -> int:
    """Calculate FLOPS for linear layers."""
    batch_size = input_shape[0]
    return batch_size * 2 * input_shape[1] * output_shape[1]

def estimate_memory_usage(shape: Tuple[int, ...], dtype: str = 'float32') -> float:
    """
    Estimate memory usage in MB for a tensor.
    """
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'float64': 8,
        'int32': 4,
        'int64': 8,
    }.get(dtype, 4)
    
    total_elements = np.prod(shape)
    memory_bytes = total_elements * bytes_per_element
    return memory_bytes / (1024 * 1024)  # Convert to MB

def format_memory(memory_mb: float) -> str:
    """Format memory size to human-readable string."""
    if memory_mb >= 1024:
        return f"{memory_mb/1024:.2f} GB"
    return f"{memory_mb:.2f} MB"

def generate_graphviz_dot(layers: List[dict]) -> str:
    """Generate GraphViz DOT representation of model architecture."""
    dot_str = ["digraph G {", "  rankdir=TB;", "  node [shape=record];"]
    
    # Add nodes
    for layer in layers:
        node_label = (f"{layer['idx']}\\n{layer['class_name']}\\n"
                     f"Params: {layer['num_params']:,}")
        dot_str.append(f'  node{layer["idx"]} [label="{node_label}"];')
    
    # Add edges
    for i in range(len(layers)-1):
        dot_str.append(f"  node{layers[i]['idx']} -> node{layers[i+1]['idx']};")
    
    dot_str.append("}")
    return "\n".join(dot_str)
