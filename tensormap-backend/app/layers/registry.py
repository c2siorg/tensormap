"""
Layer Registry for TensorMap Neural Network Builder.

This module defines the centralized registry of all supported neural network layers.
The registry pattern eliminates the need for hardcoded if/elif chains when adding
new layer types, making the system extensible and maintainable.

Architecture:
- ActivationType: Enum of supported activation functions
- ParamType: Enum of parameter data types (int, float, bool, enum)
- ParamSpec: Specification for a single layer parameter with validation constraints
- LayerSpec: Complete specification for a layer type including all parameters
- LAYER_REGISTRY: Central dictionary mapping layer type keys to their specifications

Categories:
- core: Fundamental layers (Input, Dense, Flatten)
- convolutional: Conv2D and related layers
- recurrent: LSTM, GRU, SimpleRNN
- regularization: Dropout, BatchNormalization
- pooling: MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
- encoding: Embedding layers for text/categorical data
- utility: Reshape, Concatenate (merge layer)
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActivationType(StrEnum):
    """Supported activation functions for neural network layers."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    SELU = "selu"
    ELU = "elu"
    NONE = "none"


class ParamType(StrEnum):
    """Parameter data types for layer specifications."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"


@dataclass
class ParamSpec:
    """
    Specification for a single layer parameter.

    Attributes:
        name: Parameter name (e.g., "units", "activation")
        type: Data type of the parameter
        required: Whether this parameter must be provided
        default: Default value if not provided (None for required params)
        min: Minimum value for int/float parameters
        max: Maximum value for int/float parameters
        values: List of allowed values for enum parameters
        description: Human-readable description of the parameter
    """

    name: str
    type: ParamType
    required: bool = False
    default: Any = None
    min: float | None = None
    max: float | None = None
    values: list | None = None
    description: str = ""


@dataclass
class LayerSpec:
    """
    Complete specification for a neural network layer type.

    Attributes:
        type_key: Unique identifier for this layer type (e.g., "dense", "lstm")
        display_name: Human-readable name shown in UI (e.g., "Dense", "LSTM")
        category: Layer category for UI organization
        keras_class: Fully qualified Keras class name (e.g., "tf.keras.layers.Dense")
        params: Dictionary of parameter specifications keyed by parameter name
        merge: True if this layer accepts multiple inputs (e.g., Concatenate)
        description: Human-readable description of the layer's purpose
    """

    type_key: str
    display_name: str
    category: str
    keras_class: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    merge: bool = False
    description: str = ""


# Central registry of all supported layer types
LAYER_REGISTRY: dict[str, LayerSpec] = {
    "input": LayerSpec(
        type_key="input",
        display_name="Input",
        category="core",
        keras_class="tf.keras.layers.InputLayer",
        params={
            "shape": ParamSpec(
                name="shape",
                type=ParamType.INT,
                required=True,
                description="Input dimensions e.g. 784 for flattened 28x28 images",
            )
        },
        description="Input layer defining the shape of data entering the network",
    ),
    "dense": LayerSpec(
        type_key="dense",
        display_name="Dense",
        category="core",
        keras_class="tf.keras.layers.Dense",
        params={
            "units": ParamSpec(
                name="units",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Number of neurons in the layer",
            ),
            "activation": ParamSpec(
                name="activation",
                type=ParamType.ENUM,
                values=[e.value for e in ActivationType],
                default="relu",
                description="Activation function to apply",
            ),
        },
        description="Fully connected layer with configurable neurons and activation",
    ),
    "flatten": LayerSpec(
        type_key="flatten",
        display_name="Flatten",
        category="core",
        keras_class="tf.keras.layers.Flatten",
        params={},
        description="Flattens multi-dimensional input into 1D (e.g., for transitioning from Conv2D to Dense)",
    ),
    "conv2d": LayerSpec(
        type_key="conv2d",
        display_name="Conv2D",
        category="convolutional",
        keras_class="tf.keras.layers.Conv2D",
        params={
            "filters": ParamSpec(
                name="filters",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Number of convolutional filters",
            ),
            "kernel_size": ParamSpec(
                name="kernel_size",
                type=ParamType.INT,
                min=1,
                default=3,
                description="Size of the convolutional kernel (e.g., 3 for 3x3)",
            ),
            "strides": ParamSpec(
                name="strides",
                type=ParamType.INT,
                min=1,
                default=1,
                description="Stride length for convolution",
            ),
            "padding": ParamSpec(
                name="padding",
                type=ParamType.ENUM,
                values=["valid", "same"],
                default="same",
                description="Padding mode: 'valid' (no padding) or 'same' (pad to preserve dimensions)",
            ),
            "activation": ParamSpec(
                name="activation",
                type=ParamType.ENUM,
                values=[e.value for e in ActivationType],
                default="relu",
                description="Activation function to apply",
            ),
        },
        description="2D convolutional layer for image processing",
    ),
    "maxpool2d": LayerSpec(
        type_key="maxpool2d",
        display_name="MaxPooling2D",
        category="pooling",
        keras_class="tf.keras.layers.MaxPooling2D",
        params={
            "pool_size": ParamSpec(
                name="pool_size",
                type=ParamType.INT,
                min=1,
                default=2,
                description="Size of the pooling window",
            ),
            "strides": ParamSpec(
                name="strides",
                type=ParamType.INT,
                min=1,
                default=2,
                description="Stride length for pooling",
            ),
            "padding": ParamSpec(
                name="padding",
                type=ParamType.ENUM,
                values=["valid", "same"],
                default="valid",
                description="Padding mode",
            ),
        },
        description="Max pooling layer for downsampling by taking maximum value in each window",
    ),
    "avgpool2d": LayerSpec(
        type_key="avgpool2d",
        display_name="AveragePooling2D",
        category="pooling",
        keras_class="tf.keras.layers.AveragePooling2D",
        params={
            "pool_size": ParamSpec(
                name="pool_size",
                type=ParamType.INT,
                min=1,
                default=2,
                description="Size of the pooling window",
            ),
            "strides": ParamSpec(
                name="strides",
                type=ParamType.INT,
                min=1,
                default=2,
                description="Stride length for pooling",
            ),
            "padding": ParamSpec(
                name="padding",
                type=ParamType.ENUM,
                values=["valid", "same"],
                default="valid",
                description="Padding mode",
            ),
        },
        description="Average pooling layer for downsampling by taking average value in each window",
    ),
    "globalavgpool2d": LayerSpec(
        type_key="globalavgpool2d",
        display_name="GlobalAveragePooling2D",
        category="pooling",
        keras_class="tf.keras.layers.GlobalAveragePooling2D",
        params={},
        description="Global average pooling that reduces each feature map to a single value",
    ),
    "lstm": LayerSpec(
        type_key="lstm",
        display_name="LSTM",
        category="recurrent",
        keras_class="tf.keras.layers.LSTM",
        params={
            "units": ParamSpec(
                name="units",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Number of LSTM units",
            ),
            "return_sequences": ParamSpec(
                name="return_sequences",
                type=ParamType.BOOL,
                default=False,
                description="Whether to return full sequence or just last output",
            ),
            "activation": ParamSpec(
                name="activation",
                type=ParamType.ENUM,
                values=["tanh", "sigmoid", "relu"],
                default="tanh",
                description="Activation function for the LSTM cell",
            ),
        },
        description="Long Short-Term Memory layer for sequence processing",
    ),
    "gru": LayerSpec(
        type_key="gru",
        display_name="GRU",
        category="recurrent",
        keras_class="tf.keras.layers.GRU",
        params={
            "units": ParamSpec(
                name="units",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Number of GRU units",
            ),
            "return_sequences": ParamSpec(
                name="return_sequences",
                type=ParamType.BOOL,
                default=False,
                description="Whether to return full sequence or just last output",
            ),
            "activation": ParamSpec(
                name="activation",
                type=ParamType.ENUM,
                values=["tanh", "sigmoid", "relu"],
                default="tanh",
                description="Activation function for the GRU cell",
            ),
        },
        description="Gated Recurrent Unit layer for sequence processing (simpler than LSTM)",
    ),
    "simplernn": LayerSpec(
        type_key="simplernn",
        display_name="SimpleRNN",
        category="recurrent",
        keras_class="tf.keras.layers.SimpleRNN",
        params={
            "units": ParamSpec(
                name="units",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Number of RNN units",
            ),
            "return_sequences": ParamSpec(
                name="return_sequences",
                type=ParamType.BOOL,
                default=False,
                description="Whether to return full sequence or just last output",
            ),
            "activation": ParamSpec(
                name="activation",
                type=ParamType.ENUM,
                values=["tanh", "sigmoid", "relu"],
                default="tanh",
                description="Activation function for the RNN cell",
            ),
        },
        description="Basic recurrent neural network layer for sequence processing",
    ),
    "embedding": LayerSpec(
        type_key="embedding",
        display_name="Embedding",
        category="encoding",
        keras_class="tf.keras.layers.Embedding",
        params={
            "input_dim": ParamSpec(
                name="input_dim",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Vocabulary size (number of unique tokens)",
            ),
            "output_dim": ParamSpec(
                name="output_dim",
                type=ParamType.INT,
                min=1,
                required=True,
                description="Embedding dimension (size of dense vector representation)",
            ),
        },
        description="Embedding layer for converting categorical/text data to dense vectors",
    ),
    "dropout": LayerSpec(
        type_key="dropout",
        display_name="Dropout",
        category="regularization",
        keras_class="tf.keras.layers.Dropout",
        params={
            "rate": ParamSpec(
                name="rate",
                type=ParamType.FLOAT,
                min=0.0,
                max=1.0,
                default=0.5,
                description="Fraction of inputs to drop (0.0 to 1.0)",
            )
        },
        description="Dropout layer for regularization by randomly setting inputs to zero during training",
    ),
    "batchnorm": LayerSpec(
        type_key="batchnorm",
        display_name="BatchNormalization",
        category="regularization",
        keras_class="tf.keras.layers.BatchNormalization",
        params={
            "momentum": ParamSpec(
                name="momentum",
                type=ParamType.FLOAT,
                min=0.0,
                max=1.0,
                default=0.99,
                description="Momentum for moving average",
            ),
            "epsilon": ParamSpec(
                name="epsilon",
                type=ParamType.FLOAT,
                min=1e-7,
                max=1.0,
                default=0.001,
                description="Small constant for numerical stability",
            ),
        },
        description="Batch normalization layer for normalizing activations and accelerating training",
    ),
    "reshape": LayerSpec(
        type_key="reshape",
        display_name="Reshape",
        category="utility",
        keras_class="tf.keras.layers.Reshape",
        params={
            "target_shape": ParamSpec(
                name="target_shape",
                type=ParamType.INT,
                required=True,
                description="Target shape as comma-separated integers (e.g., '7,7,64' for 7x7x64)",
            )
        },
        description="Reshape layer for changing tensor dimensions without changing data",
    ),
    "concatenate": LayerSpec(
        type_key="concatenate",
        display_name="Concatenate",
        category="utility",
        keras_class="tf.keras.layers.Concatenate",
        params={
            "axis": ParamSpec(
                name="axis",
                type=ParamType.INT,
                default=-1,
                description="Axis along which to concatenate (-1 for last axis)",
            )
        },
        merge=True,
        description="Concatenate layer for merging multiple input tensors along a specified axis",
    ),
}


def get_layer_spec(type_key: str) -> LayerSpec:
    """
    Retrieve a layer specification by its type key.

    Args:
        type_key: The unique identifier for the layer type (e.g., "dense", "lstm")

    Returns:
        LayerSpec: The complete specification for the requested layer type

    Raises:
        KeyError: If the type_key is not found in the registry
    """
    if type_key not in LAYER_REGISTRY:
        raise KeyError(f"Unknown layer type: {type_key}. Available types: {list(LAYER_REGISTRY.keys())}")
    return LAYER_REGISTRY[type_key]


def serialize_registry() -> list[dict]:
    """
    Convert the layer registry to JSON-serializable format.

    This function is used by the GET /layers API endpoint to provide
    the frontend with the complete layer registry.

    Returns:
        list[dict]: List of layer specifications as dictionaries
    """
    result = []
    for layer_spec in LAYER_REGISTRY.values():
        params_dict = {}
        for param_name, param_spec in layer_spec.params.items():
            params_dict[param_name] = {
                "name": param_spec.name,
                "type": param_spec.type.value,
                "required": param_spec.required,
                "default": param_spec.default,
                "min": param_spec.min,
                "max": param_spec.max,
                "values": param_spec.values,
                "description": param_spec.description,
            }

        result.append(
            {
                "type_key": layer_spec.type_key,
                "display_name": layer_spec.display_name,
                "category": layer_spec.category,
                "keras_class": layer_spec.keras_class,
                "params": params_dict,
                "merge": layer_spec.merge,
                "description": layer_spec.description,
            }
        )

    return result
