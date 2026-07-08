from typing import NamedTuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.modules.layer_scale import LayerScale
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.utils.config import FlowLMTransformerConfig


class KVCacheResult(NamedTuple):
    """A NamedTuple for representing the result of a key-value cache operation.
    Attributes:
    keys (torch.Tensor): The keys tensor.
    values (torch.Tensor): The values tensor.
    positions (torch.Tensor): The positions tensor.
    """
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        """Converts key-value tensors into a KVCacheResult object.
        Args:
        keys: A tensor of shape (B, H, T, D).
        values: A tensor of shape (B, H, T).
        Returns:
        KVCacheResult containing keys, values, and positions.
        """
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


def complete(
    cache: torch.Tensor, end_offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> KVCacheResult:
    """Updates the cache tensor with new key-value pairs.
    Args:
    cache (torch.Tensor): The cache to be updated.
    end_offset (torch.Tensor): The offset for indexing into the cache.
    k (torch.Tensor): The keys to add to the cache.
    v (torch.Tensor): The values to add to the cache.
    Returns:
    KVCacheResult: The updated cache and new sequence lengths.
    """
    capacity = cache.shape[3]
    assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
    B, H, T, D = k.shape
    assert T > 0
    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity
    # indexes is [B, T]
    # k is [B, H, T, D]
    # cache is [B, H, T', D]
    this_indexes = indexes.view(B, 1, T, 1)
    this_indexes = this_indexes.expand(-1, H, T, D)
    cache[0].scatter_(2, this_indexes, k)
    cache[1].scatter_(2, this_indexes, v)

    keys = cache[0]
    values = cache[1]

    indexes = torch.arange(capacity, device=end_offset.device, dtype=torch.long)

    # end_index correspond to the actual index where the last value was written.
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes - end_index

    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)
    end_offset[:] = end_offset + T
    invalid = indexes >= end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    return KVCacheResult(keys, values, positions)


class MimiStreamingMultiheadAttention(StatefulModule):
    """A class representing a stateful module for multihead attention in streaming models, utilizing Rotary Position Embeddings.
    Initialization parameters include embed_dim, num_heads, context, and rope. It defines linear projections for input and output.
    """
    def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
        """Initializes the state dictionary for transformer layers.
        Args:
        batch_size (int): The batch size of the input data.
        sequence_length (int): The length of the input sequence.
        Returns:
        dict[str, torch.Tensor]: A dictionary containing initialized tensors for the layer's state.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        out_dim = 3 * embed_dim

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        """Initializes state for attention mechanisms.
        Args:
        batch_size (int): The number of sequences in a batch.
        sequence_length (int): The length of each sequence.
        Returns:
        dict[str, torch.Tensor]: A dictionary containing 'offset', 'cache', and 'end_offset' tensors.
        """
        dim_per_head = self.embed_dim // self.num_heads

        state = {}
        state["offset"] = torch.zeros(batch_size, dtype=torch.long)
        state["cache"] = torch.zeros((2, batch_size, self.num_heads, sequence_length, dim_per_head))
        state["end_offset"] = torch.zeros(batch_size, dtype=torch.long)
        return state

    def increment_step(self, state, increment: int = 1):
        """This function increments the step in a given state by a specified increment value.
        Args:
        state (dict): The current state dictionary.
        increment (int, optional): The amount to increment the offset. Default is 1.
        Returns: None
        """
        state["offset"] += increment

    def _complete_kv(self, k, v, model_state: dict | None) -> KVCacheResult:
        """Completes a key-value pair using the provided state or initializes a new KVCacheResult.
        Args:
        k (any): The key to complete.
        v (any): The value to complete.
        model_state (dict | None): The current model state, if any.
        Returns:
        KVCacheResult: The completed key-value cache result.
        """
        if model_state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            layer_state = self.get_state(model_state)
            return complete(layer_state["cache"], layer_state["end_offset"], k, v)

    def forward(self, query: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        """Applies forward pass to query tensor using model state.
        Args:
        query (torch.Tensor): Input query tensor.
        model_state (dict | None): Optional dictionary containing model state.
        Returns:
        torch.Tensor: Result of the forward pass.
        """
        B, T = query.shape[:2]

        if model_state is None:
            offset = torch.zeros(B, device=query.device, dtype=torch.long)
        else:
            offset = self.get_state(model_state)["offset"]

        projected = self.in_proj(query)

        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

        # Permute from [b, h, t, d] to [b, t, h, d] for rope
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q, k = self.rope(q, k, offset)
        # Permute back from [b, t, h, d] to [b, h, t, d]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        k, v, pos_k = self._complete_kv(k, v, model_state)
        pos_k = pos_k[:, None]
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(
            -1, 1
        )
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        attn_bias = attn_bias & (delta < self.context)
        attn_bias = attn_bias[:, None]

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(nn.Module):
    """A class representing a streaming transformer layer that extends `nn.Module`. It includes parameters for model dimensions, attention mechanisms, feedforward layers, and other architectural details relevant to handling streaming data efficiently."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
        attention_kind: str = "mimi",
    ):
        """Initializes a transformer layer with specified parameters. Args:
        - d_model (int): The dimension of the model.
        - num_heads (int): The number of attention heads.
        - dim_feedforward (int): The dimension of the feedforward network.
        - context (int | None): Context window size for streaming multi-head attention.
        - rope (RotaryEmbedding): Rotary embedding module.
        - layer_scale (float | None): Layer scale factor, default is None.
        - attention_kind (str): Type of attention mechanism, default is "mimi". Returns: None
        """
        super().__init__()
        # Redefine self_attn to our streaming multi-head attention
        if attention_kind == "mimi":
            # TODO: we should actually use StreamingMultiheadAttention here and add context window
            # support. And we should then delete MimiStreamingMultiheadAttention.
            # The implementation is really close.
            self.self_attn = MimiStreamingMultiheadAttention(
                context=context, rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        else:
            self.self_attn = StreamingMultiheadAttention(
                rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a single transformer block operation.
        Args:
        x (torch.Tensor): Input tensor.
        model_state (dict | None): Optional model state dictionary.
        Returns:
        torch.Tensor: Output tensor after applying the transformer block.
        """
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(F.gelu(self.linear1(x)))
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        """```plaintext
        Applies a streaming transformer layer to the input tensor.
        Args:
        x (torch.Tensor): The input tensor.
        model_state (dict | None): State of the model used in attention mechanism.
        Returns:
        torch.Tensor: The output tensor after applying the streaming transformer layer.
        ```
        """
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, model_state)
        return x_orig.to(update) + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        """Applies a forward pass through the StreamingTransformer model.
        Args:
        x (torch.Tensor): Input tensor to the transformer.
        model_state (dict | None): Dictionary containing the model's state.
        Returns:
        torch.Tensor: Output tensor after applying the transformer layers.
        """
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x


class StreamingTransformer(nn.Module):
    """A transformer model for streaming data, supporting multiple heads and layers with optional layer scaling and custom dimensionality settings."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int | list[int] = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        """Initializes a new instance of the class.
        Args:
        - d_model (int): The dimensionality of the model.
        - num_heads (int): Number of attention heads.
        - num_layers (int): Number of layers in the model.
        - layer_scale (float | None, optional): Layer scaling factor. Defaults to None.
        - dim_feedforward (int | list[int], optional): Dimension of feedforward network. Defaults to 2048.
        - context (int | None, optional): Context size for attention mechanism. Defaults to None.
        - max_period (float, optional): Maximum period for positional encoding. Defaults to 10_000.0.
        - kind (str, optional): Kind of the model. Defaults to "mimi".
        Returns:
        None
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period

        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    attention_kind=kind,
                )
            )

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig) -> Self:
        """Create a FlowLMTransformer instance from Pydantic configuration.
        Args:
        config (FlowLMTransformerConfig): Configuration object containing transformer parameters.
        Returns:
        FlowLMTransformer: A new FlowLMTransformer instance configured according to the provided Pydantic config.
        """
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            kind="flow_lm",
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        """Performs a forward pass through a series of layers.
        Args:
        x (torch.Tensor): Input tensor.
        model_state (dict | None): State dictionary for the model, optional.
        Returns:
        torch.Tensor: Output tensor after processing through all layers.
        """
        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    """A class representing a Projected Transformer model, inheriting from nn.Module. It initializes a transformer with specified dimensions and parameters for handling sequential data efficiently."""
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float,
        context: int,
        max_period: float,
        dim_feedforward: int,
    ):
        """Initialize a transformer model.
        Args:
        - input_dimension (int): Dimension of the input data.
        - output_dimensions (tuple[int, ...]): Dimensions of the output layers.
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads in the model.
        - num_layers (int): Number of transformer layers.
        - layer_scale (float): Scaling factor for the transformer layers.
        - context (int): Context size for the transformer.
        - max_period (float): Maximum period for periodic functions.
        - dim_feedforward (int): Dimensionality of the feed-forward network in the model.
        Returns:
        None
        """
        super().__init__()
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
        )
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def forward(self, x, model_state: dict | None):
        """Transposes input tensor, applies projection if necessary, processes through transformer, and projects outputs before transposing again. Args: x (Tensor): Input tensor. model_state (dict | None): Optional dictionary containing model state. Returns: List of projected output tensors."""
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, model_state)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = y.transpose(1, 2)
            ys.append(y)
        return ys
