"""Configuration models for loading YAML config files."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Represents a configuration model that strictly forbids extra fields and includes specific parameters for flow-based models."""
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    """Base class for model configurations that inherit from StrictModel.
    Classes:
    FlowConfig: Configuration parameters for a flow-based model.
    FlowLMTransformerConfig: Specific configuration for a transformer used in FlowLM models.
    LookupTable: Represents a lookup table for mapping keys to values.
    """
    dim: int
    depth: int


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    """Class representing configuration for a FlowLMTransformer.
    Attributes include hidden scale, max period, model dimensions, number of heads, and layers.
    Class representing a lookup table for tokenization.
    Includes dimensions, number of bins, tokenizer type, and path to tokenizer file.
    """
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int


class LookupTable(StrictModel):
    """```python
    Class representing a lookup table with dimensions and bin settings.
    Used to store token information for natural language processing tasks.
    Configuration model for YAML config files containing data type and nested flow configuration.
    ```
    """
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str


# Root configuration
class FlowLMConfig(StrictModel):
    """Root configuration model for YAML config files."""

    dtype: str

    # Nested configurations
    flow: FlowConfig
    transformer: FlowLMTransformerConfig

    # conditioning
    lookup_table: LookupTable
    weights_path: str | None = None


# SEANet configuration
class SEANetConfig(StrictModel):
    """Class representing SEANet model configuration parameters."""
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    """MimiTransformerConfig represents the configuration for a transformer model, detailing its architecture parameters.
    QuantizerConfig specifies the configuration for quantization, including the target dimensionality.
    """
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    """```python
    Class representing configuration parameters for quantization.
    Attributes:
    dimension (int): The input dimension.
    output_dimension (int): The output dimension after quantization.
    ```
    """
    dimension: int
    output_dimension: int


# Root configuration
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    # Sample rate and channels
    sample_rate: int
    channels: int
    frame_rate: float

    # SEANet configurations
    seanet: SEANetConfig

    # Transformer
    transformer: MimiTransformerConfig

    # Quantizer
    quantizer: QuantizerConfig
    weights_path: str | None = None


class Config(StrictModel):
    """Class representing configuration settings for a model.
    Attributes include flow_lm, mimi, weights_path, and weights_path_without_voice_cloning.
    """
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    weights_path_without_voice_cloning: str | None = None


def load_config(yaml_path: str | Path) -> Config:
    """Loads a configuration from a YAML file.
    Args:
    - yaml_path (str | Path): The path to the YAML configuration file.
    Returns:
    - Config: A Config object loaded from the YAML file.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
