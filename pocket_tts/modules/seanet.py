import numpy as np
import torch.nn as nn

from .conv import StreamingConv1d, StreamingConvTranspose1d


class SEANetResnetBlock(nn.Module):
    """A class representing a residual block for a SEANet using ResNet architecture.
    Parameters:
    - dim (int): The dimension of input and output features.
    - kernel_sizes (list[int]): List of kernel sizes for convolutional layers.
    - dilations (list[int]): List of dilation rates for convolutional layers.
    - pad_mode (str): Padding mode to use ('reflect', 'replicate', etc.).
    - compress (int): Compression factor for hidden layer dimensions.
    """
    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        """Initializes a custom convolutional module with specified kernel sizes, dilations, padding mode, and compression ratio.
        Args:
        dim: The input dimension of the module.
        kernel_sizes: A list of kernel sizes for each convolutional layer.
        dilations: A list of dilation rates for each convolutional layer.
        pad_mode: Padding mode to use ("reflect", "replicate", etc.).
        compress: Compression ratio applied to the hidden dimension.
        Returns:
        None
        """
        super().__init__()
        assert len(kernel_sizes) == len(dilations), (
            "Number of kernel sizes should match number of dilations"
        )
        hidden = dim // compress
        block = nn.ModuleList([])
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                nn.ELU(alpha=1.0),
                StreamingConv1d(
                    in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, pad_mode=pad_mode
                ),
            ]
        self.block = block

    def forward(self, x, model_state: dict | None):
        """Applies a series of layers to the input tensor `x`, potentially using model state for certain layers, and returns the sum of the original and processed tensors.
        Args:
        x (Tensor): The input tensor.
        model_state (dict | None): A dictionary containing model state or None.
        Returns:
        Tensor: The output tensor after processing.
        """
        v = x
        for layer in self.block:
            if isinstance(layer, StreamingConv1d):
                v = layer(v, model_state)
            else:
                v = layer(v)
        assert x.shape == v.shape, (x.shape, v.shape, x.shape)
        return x + v


class SEANetEncoder(nn.Module):
    """Encodes input data using a series of convolutional and residual blocks.
    :param channels: Number of input channels.
    :param dimension: Feature dimension.
    :param n_filters: Number of filters in each convolutional layer.
    :param n_residual_layers: Number of residual layers.
    :param ratios: List of dilation rates for the convolutional layers.
    :param kernel_size: Kernel size for the initial convolutional layer.
    :param last_kernel_size: Kernel size for the final convolutional layer.
    :param residual_kernel_size: Kernel size for the residual blocks.
    :param dilation_base: Base value for dilation rates.
    :param pad_mode: Padding mode for convolution operations.
    :param compress: Factor for reducing output dimension.
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        """Initializes a neural network layer.
        Args:
        channels (int): Number of input and output channels.
        dimension (int): Dimensionality of the intermediate representations.
        n_filters (int): Number of filters per channel.
        n_residual_layers (int): Number of residual layers.
        ratios (list[int]): Ratios for filter size reduction.
        kernel_size (int): Size of convolutional kernels.
        last_kernel_size (int): Size of final convolutional kernel.
        residual_kernel_size (int): Size of kernels in residual blocks.
        dilation_base (int): Base value for dilation rates.
        pad_mode (str): Padding mode, default is 'reflect'.
        compress (int): Compression factor.
        Returns:
        None
        """
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks

        mult = 1
        model = nn.ModuleList(
            [StreamingConv1d(channels, mult * n_filters, kernel_size, pad_mode=pad_mode)]
        )
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                ]

            # Add downsampling layers
            model += [
                nn.ELU(alpha=1.0),
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        model += [
            nn.ELU(alpha=1.0),
            StreamingConv1d(mult * n_filters, dimension, last_kernel_size, pad_mode=pad_mode),
        ]

        self.model = model

    def forward(self, x, model_state: dict | None):
        """Applies a sequence of layers to input tensor `x`.
        Args:
        x (Tensor): Input tensor.
        model_state (dict | None): Optional dictionary containing state information for some layers.
        Returns:
        Tensor: Output tensor after passing through all layers.
        """
        for layer in self.model:
            if isinstance(layer, (StreamingConv1d, SEANetResnetBlock)):
                x = layer(x, model_state)
            else:
                x = layer(x)
        return x


class SEANetDecoder(nn.Module):
    """A class for decoding signals using a series of residual blocks and convolutional layers.
    Initializes the SEANetDecoder with parameters to control the number of channels, dimensionality, and other architectural details.
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        """Initializes a neural network layer with specified parameters.
        Args:
        - channels (int): Number of input and output channels.
        - dimension (int): Dimensionality of each channel.
        - n_filters (int): Number of filters in convolutional layers.
        - n_residual_layers (int): Number of residual blocks.
        - ratios (list[int]): Ratios for scaling dimensions.
        - kernel_size (int): Size of the main convolutional kernel.
        - last_kernel_size (int): Size of the final convolutional kernel.
        - residual_kernel_size (int): Size of the residual block kernels.
        - dilation_base (int): Base value for dilation rates.
        - pad_mode (str): Padding mode for convolution.
        - compress (int): Compression factor.
        """
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        mult = int(2 ** len(self.ratios))
        model = nn.ModuleList(
            [StreamingConv1d(dimension, mult * n_filters, kernel_size, pad_mode=pad_mode)]
        )
        # Upsample to raw audio scale
        for _, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                nn.ELU(alpha=1.0),
                StreamingConvTranspose1d(
                    mult * n_filters, mult * n_filters // 2, kernel_size=ratio * 2, stride=ratio
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            nn.ELU(alpha=1.0),
            StreamingConv1d(n_filters, channels, last_kernel_size, pad_mode=pad_mode),
        ]
        self.model = model

    def forward(self, z, model_state: dict | None):
        """Applies a sequence of layers to the input tensor `z`, optionally using a model state if provided.
        Args:
        z (Tensor): Input tensor.
        model_state (dict | None): Optional dictionary containing layer-specific state.
        Returns:
        Tensor: The output tensor after passing through all layers.
        """
        for layer in self.model:
            if isinstance(layer, (StreamingConvTranspose1d, SEANetResnetBlock, StreamingConv1d)):
                z = layer(z, model_state)
            else:
                z = layer(z)
        return z
