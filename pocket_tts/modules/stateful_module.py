from abc import ABC, abstractmethod

import torch
from torch import nn


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Initializes states for stateful modules in a given model.
    Args:
    model (nn.Module): The neural network model containing stateful modules.
    batch_size (int): The size of the input batch.
    sequence_length (int): The length of the input sequence.
    Returns:
    dict[str, dict[str, torch.Tensor]]: A dictionary mapping module names to their initialized states.
    """
    result = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module._module_absolute_name = module_name
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(
    module: nn.Module, model_state: dict[str, dict[str, torch.Tensor]], increment: int = 1
):
    """Increments the step counter of stateful modules in a model.
    Args:
    module (nn.Module): The root module to search for stateful modules.
    model_state (dict[str, dict[str, torch.Tensor]]): A dictionary containing model states by module name.
    increment (int, optional): The amount to increment each step counter. Default is 1.
    Returns:
    None
    """
    # print("incrementing steps by", increment)
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    """Abstract base class for stateful neural network modules.
    Implements initialization and retrieval of state dictionaries.
    Defines abstract method `init_state` to initialize the module's state.
    Provides methods to increment the step counter and retrieve the module's state from a model state dictionary.
    """
    def __init__(self, *args, **kwds):
        """Initialize a module with optional arguments and keyword arguments.
        Args:
        *args: Variable length argument list.
        **kwds: Arbitrary keyword arguments.
        Returns:
        None.
        """
        self._module_absolute_name = None
        return super().__init__(*args, **kwds)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize the state."""
        raise NotImplementedError

    def increment_step(self, state: dict, increment: int = 1):
        """Increment the step in the given state dictionary by a specified amount."""
        pass

    def get_state(self, model_state: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Get the state for this module from the model state."""
        return model_state[self._module_absolute_name]
