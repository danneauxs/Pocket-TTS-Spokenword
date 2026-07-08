import torch
import torch.nn as nn
from torch.nn import functional as F

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


def complete_kv(
    cache: torch.Tensor, current_end: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Updates cache tensors by appending k and v at the end.
    Args:
    cache (torch.Tensor): Cache tensor to be updated.
    current_end (torch.Tensor): Index where current data ends in cache.
    k (torch.Tensor): Tensor to append as keys.
    v (torch.Tensor): Tensor to append as values.
    Returns:
    tuple[torch.Tensor, torch.Tensor]: Updated keys and values tensors.
    """
    current_end = current_end.shape[0]

    cache[0, :, current_end : current_end + k.shape[1]] = k
    cache[1, :, current_end : current_end + v.shape[1]] = v
    valid = cache[:, :, : current_end + k.shape[1]]
    return valid[0], valid[1]


def _materialize_causal_mask(
    shape: tuple[int, ...], shift: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Create a causal mask for streaming multi-head attention.
    Args:
    shape (tuple[int, ...]): The shape of the tensor to be masked.
    shift (int): The shift value for the causal mask.
    device (str | torch.device, optional): The device on which to create the tensor. Defaults to "cpu".
    Returns:
    torch.Tensor: A causal mask tensor.
    """
    dtype = torch.float32

    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    tensor = torch.full(shape, dtype=dtype, fill_value=1, device=device)
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    mask = torch.log(mask)
    return mask.to(dtype)


class StreamingMultiheadAttention(StatefulModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        context (int, optional): Number of time steps the attention can access to.
            Can access `context` time steps into the past.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        """Initializes a transformer layer with rotary positional embeddings.
        Args:
        embed_dim (int): The embedding dimension.
        num_heads (int): Number of attention heads.
        rope (RotaryEmbedding): Rotary embedding layer.
        Returns: None
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        mult = 1
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, mult * embed_dim, bias=False)

    def _get_mask(self, shape: tuple[int, int], shift: int, device: torch.device) -> torch.Tensor:
        """Get a causal mask for the given shape and shift.
        Args:
        shape (tuple[int, int]): The shape of the mask.
        shift (int): The shift amount for the mask.
        device (torch.device): The device on which to create the mask.
        Returns:
        torch.Tensor: A causal mask tensor.
        """
        return _materialize_causal_mask(shape, shift=shift, device=device)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        """Initializes the state for a sequence processing task.
        Args:
        batch_size (int): The number of sequences in the batch.
        sequence_length (int): The length of each sequence.
        Returns:
        dict[str, torch.Tensor]: A dictionary containing the initialized state with keys 'current_end' and 'cache'.
        """
        dim_per_head = self.embed_dim // self.num_heads
        initial_current_end = torch.zeros((0,)).to(self.in_proj.weight.device)
        return dict(
            current_end=initial_current_end,
            cache=torch.full(
                (2, batch_size, sequence_length, self.num_heads, dim_per_head),
                float("NaN"),
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            ),
        )

    def increment_step(self, state: dict, increment: int = 1):
        """Increments the `current_end` size in the given state dictionary by a specified increment value.
        Args:
        state (dict): The state dictionary containing current_end tensor and other relevant information.
        increment (int, optional): The number of elements to add to current_end. Default is 1.
        Returns: None
        Completes key-value pairs using cached data and current end index in the given state dictionary.
        Args:
        k (torch.Tensor): The input query tensor.
        v (torch.Tensor): The input value tensor.
        state (dict | None): The state dictionary containing cache, current_end tensor, etc. If None, a default empty dict is used.
        Returns:
        tuple: A tuple containing the completed key and value tensors.
        Applies ROPE embeddings to query and key tensors using a streaming offset from the given state dictionary.
        Args:
        query (torch.Tensor): The input query tensor.
        key (torch.Tensor): The input key tensor.
        state (dict | None): The state dictionary containing current_end tensor, etc. If None, a default empty dict is used.
        Returns:
        tuple: A tuple containing the transformed query and key tensors.
        Retrieves the streaming offset based on the current end index in the given state dictionary.
        Args:
        state (
        """
        new_size = state["current_end"].shape[0] + increment
        state["current_end"] = torch.zeros((new_size,)).to(state["current_end"].device)

    def _complete_kv(self, k, v, state: dict | None):
        """Completes key-value pairs using cache and current end state.
        Args:
        k (any): Key.
        v (any): Value.
        state (dict | None): State dictionary containing cache and current end information.
        Returns:
        tuple: Completed key-value pairs.
        Applies rope embeddings to query and key tensors.
        Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        state (dict | None): State dictionary containing streaming offset.
        Returns:
        torch.Tensor: Embedded query tensor.
        Calculates the streaming offset from current end state.
        Args:
        state (dict | None): State dictionary containing current end information.
        Returns:
        int: Streaming offset.
        Checks if model_state is provided and raises ValueError if not.
        Args:
        model_state (dict): Model state dictionary.
        """
        k, v = complete_kv(state["cache"], state["current_end"], k, v)
        return k, v

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor, state: dict | None):
        """Apply rope embeddings to query and key tensors.
        Args:
        query (torch.Tensor): The input query tensor.
        key (torch.Tensor): The input key tensor.
        state (dict | None): The model state dictionary.
        Returns:
        torch.Tensor: The modified query and key tensors after applying rope embeddings.
        """
        # Apply rope embeddings to query and key tensors.
        streaming_offset = self._streaming_offset(state)
        return self.rope(query, key, offset=streaming_offset)

    def _streaming_offset(self, state: dict | None) -> torch.Tensor | int:
        """```python
        Returns the current streaming offset from a given state.
        Args:
        state (dict | None): The current state dictionary.
        Returns:
        torch.Tensor | int: The streaming offset.
        Checks if model_state is provided and returns the model's state.
        Args:
        model_state (dict): The model's state dictionary.
        Raises:
        ValueError: If model_state is not provided.
        Performs a forward pass through the model.
        Args:
        query (torch.Tensor): The input query tensor.
        model_state (dict | None): The current state dictionary.
        ```
        """
        return state["current_end"].shape[0]

    def check_model_state(self, model_state: dict):
        """Checks the model state and returns it if valid.
        Args:
        model_state (dict): The model's current state to check.
        Returns:
        dict: The validated model state.
        """
        if model_state is None:
            raise ValueError("model_state must be provided")
        return self.get_state(model_state)

    def forward(self, query: torch.Tensor, model_state: dict | None):
        """Calculates forward pass for a transformer model.
        Args:
        query (torch.Tensor): Input query tensor.
        model_state (dict | None): State of the model.
        Returns:
        torch.Tensor: Resulting tensor after applying attention mechanisms.
        """
        state = self.check_model_state(model_state)

        projected = self.in_proj(query)
        # Reshape from (b, t, p*h*d) to (b, t, p, h, d) where p=3, h=num_heads
        b, t, _ = projected.shape
        d = self.embed_dim // self.num_heads
        packed = projected.view(b, t, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)
        q, k = self._apply_rope(q, k, state)
        k, v = self._complete_kv(k, v, state)

        mask_shape = (query.shape[1], query.shape[1] + state["current_end"].shape[0])
        shift = state["current_end"].shape[0]

        attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2)
        # Reshape from (b, t, h, d) to (b, t, h*d)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        return x
