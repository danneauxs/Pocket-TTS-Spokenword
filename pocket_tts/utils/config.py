"""Configuration models for loading YAML config files."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Class representing a strict model configuration with forbidden extra fields."""
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    """A configuration class for managing flow dimensions and depths.
    A configuration class for transformer settings tailored for FlowLM models.
    A lookup table configuration model.
    """
    dim: int
    depth: int


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    """Class representing configuration for a language model transformer.
    Defines parameters such as hidden scale, max period, model dimensions, and layer configurations.
    Class representing a lookup table for tokenization.
    Includes settings like dimensionality, number of bins, tokenizer type, and path to tokenizer file.
    """
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int


class LookupTable(StrictModel):
    """Represents a lookup table with specified dimensions and binning."""
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
    """SEANetConfig represents the configuration parameters for a SEANet model, including dimensions, filters, and other architectural details."""
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
    """Class representing MimiTransformer model configuration parameters.
    Attributes:
    - d_model (int): Dimension of the model.
    - input_dimension (int): Input dimensionality.
    - output_dimensions (tuple[int, ...]): Output dimensions.
    - num_heads (int): Number of attention heads.
    - num_layers (int): Number of transformer layers.
    - layer_scale (float): Layer scale factor.
    - context (int): Context size for transformers.
    - max_period (float): Maximum period for positional encoding.
    - dim_feedforward (int): Dimensionality of the feed-forward network.
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
    """QuantizerConfig defines the configuration parameters for a quantization process.
    MimiConfig represents the root configuration model for Mimi YAML config files, including settings like data type and audio properties.
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
    """A class representing configuration settings for a model, including paths to weights and specific configurations for language models and voice cloning."""
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    weights_path_without_voice_cloning: str | None = None


def load_config(yaml_path: str | Path) -> Config:
    """Loads a configuration from a YAML file.
    Args:
    yaml_path (str | Path): The path to the YAML configuration file.
    Returns:
    Config: A Config object initialized with the data from the YAML file.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
