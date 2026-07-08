"""
Taken from
https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py

"""

import math

import torch
import torch.nn as nn
from typing_extensions import Self

from pocket_tts.utils.config import FlowLMConfig


def modulate(x, shift, scale):
    """modulate scales and shifts an input value.
    Args:
    x (float): Input value to be modulated.
    shift (float): Value by which to shift x.
    scale (float): Factor by which to scale x before shifting.
    Returns:
    float: Modulated value.
    rms_norm normalizes a tensor using RMS normalization.
    Args:
    x (torch.Tensor): Input tensor to normalize.
    alpha (torch.Tensor): Scaling factor for normalization.
    eps (float): Small constant to prevent division by zero.
    Returns:
    torch.Tensor: Normalized tensor.
    """
    return x * (1 + scale) + shift


def _rms_norm(x: torch.Tensor, alpha: torch.Tensor, eps: float):
    """Applies Root Mean Square Normalization to the input tensor.
    Args:
    x: Input tensor.
    alpha: Normalization factor.
    eps: Epsilon value for numerical stability.
    Returns:
    Normalized tensor.
    """
    assert x.dim() >= alpha.dim()
    x_dtype = x.dtype
    var = eps + x.var(dim=-1, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    """```python
    A custom implementation of LayerNorm for supporting jvp operations.
    ```
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """Reimplements LayerNorm to support jvp.
        Args:
        channels (int): Number of channels.
        eps (float, optional): A value added to denominator for numerical stability. Default is 1e-6.
        elementwise_affine (bool, optional): If set to True, this module has learnable affine parameters. Default is True.
        """
        super().__init__()
        self.eps = eps
        alpha_shape = (dim,)
        self.alpha = nn.Parameter(torch.full(alpha_shape, 1.0, requires_grad=True))

    def forward(self, x: torch.Tensor):
        """Applies RMS normalization to the input tensor.
        Args:
        x (torch.Tensor): Input tensor.
        Returns:
        torch.Tensor: Normalized tensor.
        """
        return _rms_norm(x, self.alpha, self.eps)


class LayerNorm(nn.Module):
    """Reimplementation of LayerNorm because the default one doesn't support jvp."""

    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        """Initializes a normalization layer for input channels.
        Args:
        channels (int): Number of channels.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-6.
        elementwise_affine (bool, optional): If True, learnable affine transformation is applied. Defaults to True.
        Returns:
        Tensor: Normalized input tensor.
        """
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """Calculates the forward pass for a normalization layer.
        Args:
        x (Tensor): Input tensor to be normalized.
        Returns:
        Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if hasattr(self, "weight"):
            x = x * self.weight + self.bias
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000
    ):
        """Initializes a neural network block with linear layers, SiLU activation, another linear layer, RMSNorm, and frequency embeddings.
        Args:
        hidden_size (int): The size of the hidden layer.
        frequency_embedding_size (int): The size of the frequency embedding. Default is 256.
        max_period (int): The maximum period for frequency embeddings. Default is 10000.
        Returns:
        None
        """
        super().__init__()
        blocks = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        blocks.append(RMSNorm(hidden_size))
        self.mlp = nn.Sequential(*blocks)
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.register_buffer(
            "freqs", torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        )

    def forward(self, t):
        """Applies a forward pass through a transformer's time embedding layer.
        Args:
        t (Tensor): Input tensor representing time.
        Returns:
        Tensor: Output tensor after applying positional encoding and MLP.
        """
        args = t * self.freqs.to(t.dtype)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        assert not (self.frequency_embedding_size % 2)
        t_emb = self.mlp(embedding)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        """Initializes a new instance of the class.
        Args:
        channels (int): The number of channels in the input and output.
        Returns:
        None
        """
        super().__init__()
        self.channels = channels

        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        """Applies an adaptive layer normalization to input and then combines it with a gated MLP output.
        Args:
        x (Tensor): Input tensor.
        y (Tensor): Modulation tensor.
        Returns:
        Tensor: Output tensor after applying the final transformations.
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        """Initializes a module for adaptive layer normalization and linear transformation.
        Args:
        model_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        Returns:
        Tensor: Transformed tensor after normalization and linear operation.
        """
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        """Computes the forward pass through an MLP with adaptive layer normalization.
        Args:
        x (Tensor): Input tensor.
        c (Tensor): Condition tensor.
        Returns:
        Tensor: Output tensor after processing through the MLP.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """Taken from https://arxiv.org/abs/2406.11838.

    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param cond_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        cond_channels,
        num_res_blocks,
        num_time_conds=1,
    ):
        """Initializes a new instance of a class.
        Args:
        in_channels (int): Number of input channels.
        model_channels (int): Number of channels in the model.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of conditional channels.
        num_res_blocks (int): Number of residual blocks.
        num_time_conds (int, optional): Number of time conditions. Default is 1.
        Returns:
        None
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        assert num_time_conds != 1
        self.time_embed = nn.ModuleList(
            [TimestepEmbedder(model_channels) for _ in range(num_time_conds)]
        )
        self.cond_embed = nn.Linear(cond_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

    @classmethod
    def from_pydantic_config(cls, cfg: FlowLMConfig, latent_dim: int, cond_dim: int) -> Self:
        """Applies a forward pass through the model. Args: c (torch.Tensor): Conditional tensor. s (torch.Tensor): Style tensor. t (torch.Tensor): Time tensor. x (torch.Tensor): Input data. Returns: torch.Tensor - Output tensor after applying the model."""
        config = cfg.flow

        flow_dim = config.dim
        flow_depth = config.depth
        num_time_conds = 2
        return SimpleMLPAdaLN(
            latent_dim, flow_dim, latent_dim, cond_dim, flow_depth, num_time_conds=num_time_conds
        )

    def forward(
        self, c: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param c: conditioning from AR transformer.
        :param s: start time tensor.
        :param t: target time tensor.
        :param x: an [N x C] Tensor of inputs.
        :return: an [N x C] Tensor of outputs.
        """
        # Combine time conditions
        ts = [s, t]
        x = self.input_proj(x)
        assert len(ts) == self.num_time_conds, (
            f"Expected {self.num_time_conds} time conditions, got {len(ts)}"
        )
        assert self.num_time_conds != 1
        t_combined = (
            sum(self.time_embed[i](ts[i]) for i in range(self.num_time_conds)) / self.num_time_conds
        )
        c = self.cond_embed(c)
        y = t_combined + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)
