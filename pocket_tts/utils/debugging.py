import torch
from torch.utils._python_dispatch import TorchDispatchMode


def to_str(obj):
    """Converts an object to a string representation.
    Args:
    obj (any): The object to convert.
    Returns:
    str: A string representation of the object.
    ```
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
        """Dispatches a PyTorch function and logs the call details.
        Args:
        func (callable): The PyTorch function to dispatch.
        types (tuple of torch.dtype): The data types of the input arguments.
        args (tuple): The positional arguments for the function.
        kwargs (dict, optional): The keyword arguments for the function.
        Returns:
        Any: The result of the function call.
        """
        output = func(*args, **kwargs or {})
        print(
            f"Aten function called: {func}, args: "
            f"{to_str(args)}, kwargs: {to_str(kwargs)} -> "
            f"output: {to_str(output)}"
        )
        return output
