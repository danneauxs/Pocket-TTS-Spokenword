from abc import ABC, abstractmethod

import torch
from torch import nn


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Initialize state dictionaries for each stateful module in a model.
    Args:
    - model (nn.Module): The neural network model containing modules to initialize.
    - batch_size (int): The size of the input batch.
    - sequence_length (int): The length of the input sequence.
    Returns:
    - dict[str, dict[str, torch.Tensor]]: A dictionary mapping module names to their state dictionaries.
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
    """Increments the step counter of each stateful module in a given model.
    Args:
    module (nn.Module): The root module to search for stateful modules.
    model_state (dict[str, dict[str, torch.Tensor]]): Dictionary containing module names and their corresponding states.
    increment (int, optional): The amount by which to increment the step counter. Defaults to 1.
    Returns:
    None
    """
    # print("incrementing steps by", increment)
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    """A base class for modules that maintain internal state across multiple inference steps.
    Abstract methods:
    - init_state: Initializes the internal state for a given batch size and sequence length.
    - get_state: Retrieves the current state of the module from a model's state dictionary.
    """
    def __init__(self, *args, **kwds):
        """Initializes the module and calls the superclass constructor.
        Args:
        *args: Positional arguments to pass to the superclass constructor.
        **kwds: Keyword arguments to pass to the superclass constructor.
        Abstract method to initialize the state of the module.
        Args:
        batch_size (int): The size of the input batch.
        sequence_length (int): The length of the input sequence.
        Increments the step count in the given state dictionary by a specified increment.
        Args:
        state (dict): The state dictionary containing the step count.
        increment (int, optional): The value to add to the step count. Defaults to 1.
        Retrieves the state for this module from the provided model state.
        Args:
        model_state (dict[str, dict[str, torch.Tensor]]): A dictionary mapping module names to their states.
        Returns:
        dict[str, torch.Tensor]: The state for this module.
        """
        self._module_absolute_name = None
        return super().__init__(*args, **kwds)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize the state."""
        raise NotImplementedError

    def increment_step(self, state: dict, increment: int = 1):
        """Increment the step in the given state by a specified increment.
        Args:
        state (dict): The dictionary containing the current state.
        increment (int, optional): The amount to increment the step by. Defaults to 1.
        Returns:
        dict: The updated state with the incremented step.
        Retrieve the state for this module from the model state.
        Args:
        model_state (dict[str, dict[str, torch.Tensor]]): The dictionary containing all module states.
        Returns:
        dict[str, torch.Tensor]: The state of this module.
        """
        pass

    def get_state(self, model_state: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Get the state for this module from the model state."""
        return model_state[self._module_absolute_name]
