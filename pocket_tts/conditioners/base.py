import logging
from typing import Generic, NamedTuple, TypeVar

import torch
from torch import nn

logger = logging.getLogger(__name__)


Prepared = TypeVar("Prepared")  # represents the prepared condition input type.


class TokenizedText(NamedTuple):
    """A named tuple representing tokenized text as a long tensor."""
    tokens: torch.Tensor  # should be long tensor.


class BaseConditioner(nn.Module, Generic[Prepared]):
    """Base model for all conditioner modules.

    Args:
        dim (int): internal dim of the model.
        output_dim (int): Output dim of the conditioner.
        force_linear (bool, optional): Force linear projection even when `dim == output_dim`.
        output_bias (bool): if True, the output projection will have a bias.
        learn_padding (bool): if True, the padding value will be learnt, zero otherwise.
    """

    def __init__(
        self, dim: int, output_dim: int, output_bias: bool = False, force_linear: bool = True
    ):
        """Initialize a transformation layer.
        Args:
        dim (int): Input dimension.
        output_dim (int): Output dimension.
        output_bias (bool, optional): Whether to include bias. Defaults to False.
        force_linear (bool, optional): Force linear transformation if dimensions don't match. Defaults to True.
        Returns:
        torch.Tensor: Transformed output tensor.
        """
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        assert force_linear or dim != output_dim
        assert not output_bias

    def forward(self, inputs: TokenizedText) -> torch.Tensor:
        """Executes a forward pass through the network using the provided tokenized text inputs.
        Args:
        inputs (TokenizedText): The input data containing tokenized text.
        Returns:
        torch.Tensor: The output tensor from the forward pass.
        """
        return self._get_condition(inputs)
