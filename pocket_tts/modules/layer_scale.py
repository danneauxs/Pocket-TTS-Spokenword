import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """A LayerScale module scales its input tensor by learnable per-channel weights initialized to a specified value."""
    def __init__(self, channels: int, init: float):
        """Initializes a scaling layer.
        Args:
        - channels (int): Number of channels.
        - init (float): Initial value for scaling parameters.
        Returns:
        - torch.Tensor: Scaled input tensor.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor):
        """Applies a scaling factor to input tensor.
        Args:
        x (torch.Tensor): Input tensor.
        Returns:
        torch.Tensor: Scaled output tensor.
        """
        return self.scale * x
