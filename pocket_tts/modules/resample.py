import torch
from torch import nn

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    """

    def __init__(self, stride: int, dimension: int):
        """Initializes a custom convolutional layer for processing sequential data.
        Args:
        stride (int): Stride of the convolution.
        dimension (int): Dimensionality of the input and output.
        Returns:
        torch.Tensor: Output tensor after applying the convolution.
        """
        super().__init__()
        self.conv = StreamingConv1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        """Forward pass through transposed convolutional layer to upsample input tensor `x` by an integer factor defined in `model_state`.
        Args:
        x (torch.Tensor): Input tensor.
        model_state (dict | None): Dictionary containing model state, including stride information.
        Returns:
        torch.Tensor: Upsampled output tensor.
        """
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(self, stride: int, dimension: int):
        """Initializes a transposed convolution layer for processing time-series data.
        Args:
        stride (int): The stride of the convolution.
        dimension (int): The number of input and output channels.
        Returns:
        torch.Tensor: The result of the transposed convolution operation.
        """
        super().__init__()
        self.convtr = StreamingConvTranspose1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        """Performs a forward pass through the transformer layer using input tensor `x` and optional `model_state`.
        Args:
        x (torch.Tensor): Input tensor.
        model_state (dict | None): Optional dictionary containing model state.
        Returns:
        torch.Tensor: Output tensor after applying the transformer layer.
        """
        return self.convtr(x, model_state)
