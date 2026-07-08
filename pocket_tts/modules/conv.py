import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from pocket_tts.modules.stateful_module import StatefulModule


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


class StreamingConv1d(StatefulModule):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        """Initialize a custom convolutional layer.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        dilation (int, optional): Dilation rate of the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        bias (bool, optional): Whether to include a bias term. Default is True.
        pad_mode (str, optional): Mode of padding. Can be 'constant' or 'replicate'. Default is 'constant'.
        Returns:
        None
        """
        super().__init__()
        assert pad_mode in ["constant", "replicate"], pad_mode
        self.pad_mode = pad_mode
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @property
    def _stride(self) -> int:
        """Initialize state for convolutional layer.
        Args:
        batch_size (int): Number of samples in a batch.
        sequence_length (int): Length of input sequence.
        Returns:
        dict[str, torch.Tensor]: Dictionary containing initial state tensors.
        """
        return self.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        """Returns the effective kernel size of the convolutional layer, considering dilation.
        Args:
        batch_size (int): The batch size for the initial state.
        sequence_length (int): The length of the sequence for the initial state.
        Returns:
        dict[str, torch.Tensor]: A dictionary containing the previous hidden state and a boolean tensor indicating the first step.
        """
        return self.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        """Computes the effective kernel size of a convolutional layer considering dilation.
        Args:
        self: The instance of the class containing the convolutional layer.
        Returns: An integer representing the effective kernel size.
        """
        dilation = self.conv.dilation[0]
        return (self._kernel_size - 1) * dilation + 1  # effective kernel size with dilations

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        """Initializes the state for processing a batch of sequences.
        Args:
        batch_size (int): Number of items in the batch.
        sequence_length (int): Length of the input sequence.
        Returns:
        dict[str, torch.Tensor]: Dictionary containing initial state tensors.
        """
        stride = self._stride
        # Effective kernel size accounting for dilation.
        kernel = self._effective_kernel_size
        previous = torch.zeros(batch_size, self.conv.in_channels, kernel - stride)
        first = torch.ones(batch_size, dtype=torch.bool)
        return dict(previous=previous, first=first)

    def forward(self, x, model_state: dict | None):
        """Applies a forward pass through a streaming model.
        Args:
        x (Tensor): Input tensor of shape [B, C, T].
        model_state (dict | None): State dictionary for streaming inference, or None if starting from scratch.
        Returns:
        Tensor: Output tensor after processing the input.
        """
        B, C, T = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self.get_state(model_state)
        TP = state["previous"].shape[-1]
        if TP and self.pad_mode == "replicate":
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            state["previous"][:] = torch.where(
                state["first"].view(-1, 1, 1), init, state["previous"]
            )
        if TP:
            x = torch.cat([state["previous"], x], dim=-1)
        y = self.conv(x)
        if TP:
            state["previous"][:] = x[..., -TP:]
            if self.pad_mode == "replicate":
                state["first"] = torch.zeros_like(state["first"])
        return y


class StreamingConvTranspose1d(StatefulModule):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """Initializes a 1D transposed convolution layer.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        Returns:
        None
        """
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )

    @property
    def _stride(self) -> int:
        """Returns a dictionary containing a tensor for partial state.
        Args:
        batch_size (int): The batch size.
        sequence_length (int): The sequence length.
        Returns:
        dict[str, torch.Tensor]: A dictionary with a single key "partial" mapping to a tensor of zeros.
        """
        return self.convtr.stride[0]

    @property
    def _kernel_size(self) -> int:
        """Returns the kernel size of the convolutional transformer.
        Args:
        batch_size (int): The batch size for the state.
        sequence_length (int): The length of the input sequence.
        Returns:
        dict[str, torch.Tensor]: A dictionary containing the initial state with a "partial" key.
        """
        return self.convtr.kernel_size[0]

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        """Initializes state for convolutional transformation.
        Args:
        batch_size (int): The number of items in a batch.
        sequence_length (int): Length of input sequences.
        Returns:
        dict[str, torch.Tensor]: A dictionary containing initial state with 'partial' key.
        """
        K = self._kernel_size
        S = self._stride
        return dict(partial=torch.zeros(batch_size, self.convtr.out_channels, K - S))

    def forward(self, x, mimi_state: dict):
        """Computes the forward pass of a convolutional layer with partial state management.
        Args:
        x (Tensor): Input tensor.
        mimi_state (dict): Dictionary containing layer state information.
        Returns:
        Tensor: Output tensor after applying the convolution and partial state updates.
        """
        layer_state = self.get_state(mimi_state)["partial"]
        y = self.convtr(x)
        PT = layer_state.shape[-1]
        if PT > 0:
            y[..., :PT] += layer_state
            bias = self.convtr.bias
            for_partial = y[..., -PT:]
            if bias is not None:
                for_partial -= bias[:, None]
            layer_state[:] = for_partial
            y = y[..., :-PT]
        return y
