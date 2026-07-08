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
    """Create a cache directory for pocket_tts.
    Args:
    None
    Returns:
    Path object representing the created cache directory
    """
    cache_dir = Path.home() / ".cache" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def print_nb_parameters(model: nn.Module, model_name: str):
    """Counts and logs the number of parameters in a PyTorch model.
    Args:
    model (nn.Module): The PyTorch model whose parameters are to be counted.
    model_name (str): The name of the model for logging purposes.
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
    """Calculates the total size of tensors in a nested dictionary.
    Args:
    state_dict (dict): The nested dictionary containing tensors and other dictionaries.
    Returns:
    int: Total size of tensors in bytes.
    ---
    Tracks and optionally prints the execution time of a task.
    Args:
    task_name (str): Name of the task being timed.
    print_output (bool, optional): Whether to print the execution time. Defaults to True.
    Methods:
    start(): Marks the beginning of the task.
    end(): Marks the end of the task and prints or returns the elapsed time.
    """
    total_size = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()
        elif isinstance(value, dict):
            total_size += size_of_dict(value)
    return total_size


class display_execution_time:
    """A context manager for displaying the execution time of a task.
    Tracks the start and end times to calculate the elapsed duration in milliseconds.
    Optionally prints the output.
    """
    def __init__(self, task_name: str, print_output: bool = True):
        """Measures and prints elapsed time for a task.
        Args:
        task_name (str): Name of the task being measured.
        print_output (bool, optional): Whether to print the elapsed time. Defaults to True.
        Returns:
        None
        """
        self.task_name = task_name
        self.print_output = print_output
        self.start_time = None
        self.elapsed_time_ms = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """This context manager measures and logs the time taken for a task.
        Args:
        self: The instance of the context manager.
        Returns:
        self
        Function to download a file if it's not already cached. If the file path is a URL, it will be downloaded and stored in a cache directory.
        Args:
        file_path (str): The path to the file or URL.
        Returns:
        Path: The path to the local copy of the file.
        ```
        """
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the context manager and logs the elapsed time.
        Args:
        exc_type (type): The exception type.
        exc_val (Exception): The exception value.
        exc_tb (traceback): The traceback object.
        Returns:
        bool: False to not suppress exceptions.
        """
        end_time = time.monotonic()
        self.elapsed_time_ms = int((end_time - self.start_time) * 1000)
        if self.print_output:
            self.logger.info("%s took %d ms", self.task_name, self.elapsed_time_ms)
        return False  # Don't suppress exceptions


def download_if_necessary(file_path: str) -> Path:
    """Download a file if it's not already cached.
    Args:
    file_path (str): The URL or local path of the file to download.
    Returns:
    Path: The path to the locally cached file.
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
    """Loads a predefined voice as a torch.Tensor.
    Args:
    - voice_name (str): The name of the predefined voice to load.
    Returns:
    - torch.Tensor: The loaded voice tensor.
    """
    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found"
            f", available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    # There is only one tensor in the file.
    return safetensors.torch.load_file(voice_file)["audio_prompt"]
