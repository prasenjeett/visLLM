import pytest
import torch
import torch.nn as nn
from torch_model_analyzer import model_summary, FormattingOptions

def test_basic_model_summary():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    
    summary = model_summary(model, input_size=(3, 32, 32))
    assert isinstance(summary, str)
    assert "Model Summary" in summary

def test_memory_estimation():
    model = nn.Linear(100, 10)
    summary = model_summary(
        model, 
        input_size=(100,),
        formatting_options=FormattingOptions(show_memory=True)
    )
    assert "Memory (MB)" in summary

def test_flops_calculation():
    model = nn.Conv2d(3, 64, 3)
    summary = model_summary(
        model,
        input_size=(3, 32, 32),
        formatting_options=FormattingOptions(show_flops=True)
    )
    assert "FLOPS" in summary
