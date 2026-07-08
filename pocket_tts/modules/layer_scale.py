import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """A LayerScale module for scaling layer outputs in neural networks. Applies a learnable scale factor to the input tensor along its channel dimension."""
    def __init__(self, channels: int, init: float):
        """Initializes a scaling layer for input tensors.
        Args:
        channels (int): Number of channels in the input tensor.
        init (float): Initial value for the scaling parameter.
        Returns:
        torch.Tensor: Scaled output tensor.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor):
        """Applies a scaling factor to the input tensor.
        Args:
        x (torch.Tensor): The input tensor.
        Returns:
        torch.Tensor: The scaled tensor.
        """
        return self.scale * x
