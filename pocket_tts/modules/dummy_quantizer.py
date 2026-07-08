import torch
from torch import nn


class DummyQuantizer(nn.Module):
    """Simplified quantizer that only provides output projection for TTS.

    This removes all unnecessary quantization logic since we don't use actual quantization.
    """

    def __init__(self, dimension: int, output_dimension: int):
        """Initializes a convolutional layer for projecting input tensors.
        Args:
        dimension (int): The input tensor's channel dimension.
        output_dimension (int): The output tensor's channel dimension.
        Returns:
        torch.Tensor: The projected tensor after convolution.
        """
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a linear transformation to the input tensor `x`.
        Args:
        - x (torch.Tensor): The input tensor.
        Returns:
        - torch.Tensor: The transformed tensor.
        """
        return self.output_proj(x)
