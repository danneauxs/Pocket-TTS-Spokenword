import hashlib
import logging
import time
from pathlib import Path

import requests
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from torch import nn

PROJECT_ROOT = Path(__file__).parent.parent.parent

_voices_names = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
PREDEFINED_VOICES = {
    # don't forget to change this
    x: f"hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{x}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    for x in _voices_names
}


def make_cache_directory() -> Path:
    """Create a cache directory for the application.
    Args:
    - None
    Returns:
    - Path: The path to the created or existing cache directory.
    Log and print the number of parameters in a given neural network model.
    Args:
    - model (nn.Module): The neural network model.
    - model_name (str): The name of the model.
    Returns:
    - None
    """
    cache_dir = Path.home() / ".cache" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def print_nb_parameters(model: nn.Module, model_name: str):
    """Logs the number of parameters in a model and returns the total count.
    Args:
    model (nn.Module): The neural network model.
    model_name (str): Name of the model for logging purposes.
    Returns:
    int: Total number of parameters in the model.
    """
    logger = logging.getLogger(__name__)
    state_dict = model.state_dict()
    total = 0
    for key, value in state_dict.items():
        logger.info("%s: %,d", key, value.numel())
        total += value.numel()
    logger.info("Total number of parameters in %s: %,d", model_name, total)


def size_of_dict(state_dict: dict) -> int:
    """Calculates the total size of a dictionary containing tensors and nested dictionaries.
    Args:
    state_dict (dict): Dictionary to calculate the size of.
    Returns:
    int: Total size in bytes.
    ---
    Decorator class to display execution time for tasks.
    """
    total_size = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()
        elif isinstance(value, dict):
            total_size += size_of_dict(value)
    return total_size


class display_execution_time:
    """Tracks and optionally prints the execution time of a task.
    Context manager for measuring and logging the duration of a code block. Logs the elapsed time in milliseconds when exiting the context.
    """
    def __init__(self, task_name: str, print_output: bool = True):
        """Context manager for timing and optionally logging tasks.
        Args:
        task_name (str): Name of the task being timed.
        print_output (bool, optional): Whether to print elapsed time upon exit. Default is True.
        Returns:
        None
        """
        self.task_name = task_name
        self.print_output = print_output
        self.start_time = None
        self.elapsed_time_ms = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Context manager for timing operations.
        Args:
        task_name (str): Name of the task to be logged.
        print_output (bool): Whether to print the elapsed time.
        logger (Logger): Logger object for logging messages.
        Returns:
        None
        """
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes a timing context manager and logs the elapsed time.
        Args:
        exc_type: The type of exception raised.
        exc_val: The value of the exception raised.
        exc_tb: The traceback object if an exception was raised.
        Returns:
        False to indicate that exceptions should not be suppressed.
        """
        end_time = time.monotonic()
        self.elapsed_time_ms = int((end_time - self.start_time) * 1000)
        if self.print_output:
            self.logger.info("%s took %d ms", self.task_name, self.elapsed_time_ms)
        return False  # Don't suppress exceptions


def download_if_necessary(file_path: str) -> Path:
    """Downloads a file from a given URL if it's not already cached and returns the local path to the file.
    Args:
    file_path (str): The URL of the file to download or the hf:// path of the file.
    Returns:
    Path: The local path to the downloaded or cached file.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        cache_dir = make_cache_directory()
        cached_file = cache_dir / (
            hashlib.sha256(file_path.encode()).hexdigest() + "." + file_path.split(".")[-1]
        )
        if not cached_file.exists():
            response = requests.get(file_path)
            response.raise_for_status()
            with open(cached_file, "wb") as f:
                f.write(response.content)
        return cached_file
    elif file_path.startswith("hf://"):
        file_path = file_path.removeprefix("hf://")
        splitted = file_path.split("/")
        repo_id = "/".join(splitted[:2])
        filename = "/".join(splitted[2:])
        if "@" in filename:
            filename, revision = filename.split("@")
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    else:
        return Path(file_path)


def load_predefined_voice(voice_name: str) -> torch.Tensor:
    """Loads a predefined voice from a file.
    Args:
    voice_name (str): The name of the predefined voice to load.
    Returns:
    torch.Tensor: A tensor containing the audio prompt for the specified voice.
    Raises:
    ValueError: If the voice_name is not found in PREDEFINED_VOICES.
    """
    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found"
            f", available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    # There is only one tensor in the file.
    return safetensors.torch.load_file(voice_file)["audio_prompt"]
