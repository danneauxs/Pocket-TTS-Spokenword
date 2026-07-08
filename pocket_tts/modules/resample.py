import torch
from torch import nn

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    """

    def __init__(self, stride: int, dimension: int):
        """Initialize a StreamingConv1d layer for processing sequences.
        Args:
        stride (int): Stride of the convolution.
        dimension (int): Dimensionality of the input and output space.
        Returns:
        torch.Tensor: The result of applying the convolution to the input tensor.
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
        """Upsample a 1D tensor by an integer factor using transposed convolutions.
        Args:
        x (torch.Tensor): Input tensor to be upsampled.
        model_state (dict | None): Optional dictionary containing model state information.
        Returns:
        torch.Tensor: Upsampled tensor.
        """
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(self, stride: int, dimension: int):
        """Initializes a transposed convolutional layer for streaming data.
        Args:
        stride (int): The stride of the convolution.
        dimension (int): The number of input and output channels.
        Returns:
        torch.Tensor: The output tensor after applying the transposed convolution.
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
        """Applies a convolutional transformation to the input tensor `x` using the provided `model_state`.
        Args:
        x (torch.Tensor): The input tensor to be transformed.
        model_state (dict | None): A dictionary containing the state of the model.
        Returns:
        torch.Tensor: The transformed output tensor.
        """
        return self.convtr(x, model_state)
