import numpy as np
import torch.nn as nn

from .conv import StreamingConv1d, StreamingConvTranspose1d


class SEANetResnetBlock(nn.Module):
    """A custom residual block for SEANet using a ResNet architecture with variable kernel sizes and dilations."""
    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        """Initializes a custom module with specified dimensions and kernel parameters.
        Args:
        dim: The input dimension.
        kernel_sizes: List of kernel sizes for each convolutional block.
        dilations: List of dilation rates for each convolutional block.
        pad_mode: Padding mode to use, default is "reflect".
        compress: Compression factor for hidden layer size.
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
        """Computes the forward pass of an SEANet encoder.
        Args:
        x (Tensor): Input tensor.
        model_state (dict | None): Optional dictionary containing model state for StreamingConv1d layers.
        Returns:
        Tensor: Output tensor after processing through the network.
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
    """A class for constructing a SEANetEncoder neural network module."""
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
        channels (int): Number of input and output channels.
        dimension (int): Dimensionality of the input and output features.
        n_filters (int): Number of filters in convolutional layers.
        n_residual_layers (int): Number of residual blocks.
        ratios (list[int]): Ratios for different filter sizes.
        kernel_size (int): Size of the main convolutional kernel.
        last_kernel_size (int): Size of the final convolutional kernel.
        residual_kernel_size (int): Size of the kernels in residual blocks.
        dilation_base (int): Base value for dilation rates.
        pad_mode (str): Padding mode to use during convolution.
        compress (int): Compression factor for output channels.
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
        """Applies a sequence of layers to the input tensor `x`. If a layer is an instance of StreamingConv1d or SEANetResnetBlock, it passes the model state to the layer.
        Args:
        x (Tensor): Input tensor.
        model_state (dict | None): Dictionary containing model state. Optional.
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
    """SEANetDecoder class implements a decoder for SEANet architecture, designed to process input data through multiple residual layers and filters to reconstruct output."""
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
        """Initialize a neural network model.
        Args:
        channels (int): Number of input channels.
        dimension (int): Dimension of each channel.
        n_filters (int): Number of filters in convolutional layers.
        n_residual_layers (int): Number of residual layers.
        ratios (list[int]): Ratios for different parts of the model.
        kernel_size (int): Size of the convolution kernel.
        last_kernel_size (int): Size of the final convolution kernel.
        residual_kernel_size (int): Size of the residual block kernel.
        dilation_base (int): Base for dilation in convolution layers.
        pad_mode (str): Padding mode, default is 'reflect'.
        compress (int): Compression factor for some layers.
        Returns:
        None
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
        """Applies a sequence of layers to input `z`, optionally using state from `model_state`. Args: z (tensor): Input tensor. model_state (dict | None): Optional dictionary containing layer-specific state. Returns: Transformed output tensor after passing through all layers."""
        for layer in self.model:
            if isinstance(layer, (StreamingConvTranspose1d, SEANetResnetBlock, StreamingConv1d)):
                z = layer(z, model_state)
            else:
                z = layer(z)
        return z
