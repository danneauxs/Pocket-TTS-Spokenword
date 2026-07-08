import torch
from torch import nn


class DummyQuantizer(nn.Module):
    """Simplified quantizer that only provides output projection for TTS.

    This removes all unnecessary quantization logic since we don't use actual quantization.
    """

    def __init__(self, dimension: int, output_dimension: int):
        """Initializes a layer for projecting input tensors along their dimension using a convolutional operation.
        Args:
        dimension (int): The input and output feature dimensions.
        output_dimension (int): The desired output feature dimension after projection.
        Returns:
        torch.Tensor: The projected tensor with the specified output dimension.
        """
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a linear projection to the input tensor.
        Args:
        x (torch.Tensor): The input tensor.
        Returns:
        torch.Tensor: The projected output tensor.
        """
        return self.output_proj(x)
