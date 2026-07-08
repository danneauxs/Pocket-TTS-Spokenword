import torch
from torch.utils._python_dispatch import TorchDispatchMode


def to_str(obj):
    """Converts an object to a string representation.
    Args:
    obj (any): The object to convert.
    Returns:
    str: A string representation of the object.
    """
    if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
        return f"T(s={list(obj.shape)})"
    elif isinstance(obj, (list, tuple)):
        return "[" + ", ".join(to_str(o) for o in obj) + "]"
    elif isinstance(obj, dict):
        return "{" + ", ".join(f"{to_str(k)}: {to_str(v)}" for k, v in obj.items()) + "}"
    else:
        return str(obj)


class LoggingMode(TorchDispatchMode):
    """Useful to check implementation differences."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Dispatches a PyTorch function and logs its details.
        Args:
        func (callable): The function to be called.
        types (tuple): Types of arguments.
        args (tuple, optional): Positional arguments for the function. Defaults to an empty tuple.
        kwargs (dict, optional): Keyword arguments for the function. Defaults to None.
        Returns:
        Any: The result of calling the function with the provided arguments.
        """
        output = func(*args, **kwargs or {})
        print(
            f"Aten function called: {func}, args: "
            f"{to_str(args)}, kwargs: {to_str(kwargs)} -> "
            f"output: {to_str(output)}"
        )
        return output
