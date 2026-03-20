"""Layer registry — the single source of truth for all supported layer types.

To add a new layer:
1. Write a builder function _build_<n>(params, input_tensor, name) -> tensor
2. Add a LayerSpec entry to LAYER_REGISTRY.
That's it. No other files need to be changed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LayerSpec:
    node_type: str
    display_name: str
    builder: Callable[[dict, Any, str], Any]
    default_params: dict = field(default_factory=dict)
    description: str = ""
    category: str = "core"


def _build_dense(params, input_tensor, name):
    activation = params.get("activation", "relu")
    return tf.keras.layers.Dense(
        units=int(params["units"]),
        activation="linear" if activation == "none" else activation,
        name=name,
    )(input_tensor)


def _build_flatten(params, input_tensor, name):
    return tf.keras.layers.Flatten(name=name)(input_tensor)


def _build_conv2d(params, input_tensor, name):
    activation = params.get("activation", "relu")
    return tf.keras.layers.Conv2D(
        filters=int(params["filter"]),
        kernel_size=(int(params["kernelX"]), int(params["kernelY"])),
        strides=(int(params["strideX"]), int(params["strideY"])),
        padding=params.get("padding", "valid"),
        activation="linear" if activation == "none" else activation,
        name=name,
    )(input_tensor)


def _build_maxpooling2d(params, input_tensor, name):
    return tf.keras.layers.MaxPooling2D(
        pool_size=(int(params.get("poolX", 2)), int(params.get("poolY", 2))),
        strides=(int(params.get("strideX", 2)), int(params.get("strideY", 2))),
        padding=params.get("padding", "valid"),
        name=name,
    )(input_tensor)


def _build_avgpooling2d(params, input_tensor, name):
    return tf.keras.layers.AveragePooling2D(
        pool_size=(int(params.get("poolX", 2)), int(params.get("poolY", 2))),
        strides=(int(params.get("strideX", 2)), int(params.get("strideY", 2))),
        padding=params.get("padding", "valid"),
        name=name,
    )(input_tensor)


def _build_globalmaxpooling2d(params, input_tensor, name):
    return tf.keras.layers.GlobalMaxPooling2D(name=name)(input_tensor)


def _build_globalavgpooling2d(params, input_tensor, name):
    return tf.keras.layers.GlobalAveragePooling2D(name=name)(input_tensor)


def _build_dropout(params, input_tensor, name):
    return tf.keras.layers.Dropout(
        rate=float(params.get("rate", 0.5)),
        name=name,
    )(input_tensor)


def _build_batchnorm(params, input_tensor, name):
    return tf.keras.layers.BatchNormalization(
        momentum=float(params.get("momentum", 0.99)),
        epsilon=float(params.get("epsilon", 0.001)),
        name=name,
    )(input_tensor)


def _build_layernorm(params, input_tensor, name):
    return tf.keras.layers.LayerNormalization(
        epsilon=float(params.get("epsilon", 0.001)),
        name=name,
    )(input_tensor)


def _build_lstm(params, input_tensor, name):
    return tf.keras.layers.LSTM(
        units=int(params.get("units", 64)),
        return_sequences=str(params.get("return_sequences", "false")).lower() == "true",
        dropout=float(params.get("dropout", 0.0)),
        name=name,
    )(input_tensor)


def _build_gru(params, input_tensor, name):
    return tf.keras.layers.GRU(
        units=int(params.get("units", 64)),
        return_sequences=str(params.get("return_sequences", "false")).lower() == "true",
        dropout=float(params.get("dropout", 0.0)),
        name=name,
    )(input_tensor)


def _build_simplernn(params, input_tensor, name):
    return tf.keras.layers.SimpleRNN(
        units=int(params.get("units", 32)),
        return_sequences=str(params.get("return_sequences", "false")).lower() == "true",
        name=name,
    )(input_tensor)


def _build_reshape(params, input_tensor, name):
    raw = params.get("target_shape", "")
    target_shape = tuple(int(d.strip()) for d in raw.split(",") if d.strip())
    return tf.keras.layers.Reshape(target_shape=target_shape, name=name)(input_tensor)


def _build_embedding(params, input_tensor, name):
    return tf.keras.layers.Embedding(
        input_dim=int(params.get("input_dim", 1000)),
        output_dim=int(params.get("output_dim", 64)),
        name=name,
    )(input_tensor)


LAYER_REGISTRY: dict[str, LayerSpec] = {
    "custominput": LayerSpec(
        node_type="custominput",
        display_name="Input",
        builder=None,
        default_params={"dim-1": 28, "dim-2": 28, "dim-3": 1},
        description="Network input — define the shape of one input tensor.",
        category="core",
    ),
    "customdense": LayerSpec(
        node_type="customdense",
        display_name="Dense",
        builder=_build_dense,
        default_params={"units": 64, "activation": "relu"},
        description="Fully-connected (Dense) layer.",
        category="core",
    ),
    "customflatten": LayerSpec(
        node_type="customflatten",
        display_name="Flatten",
        builder=_build_flatten,
        default_params={},
        description="Flatten all dimensions except batch into a 1-D vector.",
        category="core",
    ),
    "customconv": LayerSpec(
        node_type="customconv",
        display_name="Conv2D",
        builder=_build_conv2d,
        default_params={
            "filter": 32,
            "kernelX": 3,
            "kernelY": 3,
            "strideX": 1,
            "strideY": 1,
            "padding": "valid",
            "activation": "relu",
        },
        description="2-D convolution layer.",
        category="core",
    ),
    "custommaxpool2d": LayerSpec(
        node_type="custommaxpool2d",
        display_name="MaxPooling2D",
        builder=_build_maxpooling2d,
        default_params={"poolX": 2, "poolY": 2, "strideX": 2, "strideY": 2, "padding": "valid"},
        description="Max-pooling over 2-D spatial data.",
        category="pooling",
    ),
    "customavgpool2d": LayerSpec(
        node_type="customavgpool2d",
        display_name="AvgPooling2D",
        builder=_build_avgpooling2d,
        default_params={"poolX": 2, "poolY": 2, "strideX": 2, "strideY": 2, "padding": "valid"},
        description="Average-pooling over 2-D spatial data.",
        category="pooling",
    ),
    "customglobalmaxpool2d": LayerSpec(
        node_type="customglobalmaxpool2d",
        display_name="GlobalMaxPooling2D",
        builder=_build_globalmaxpooling2d,
        default_params={},
        description="Global max pooling — reduces each feature map to a scalar.",
        category="pooling",
    ),
    "customglobalavgpool2d": LayerSpec(
        node_type="customglobalavgpool2d",
        display_name="GlobalAvgPooling2D",
        builder=_build_globalavgpooling2d,
        default_params={},
        description="Global average pooling — reduces each feature map to a scalar.",
        category="pooling",
    ),
    "customdropout": LayerSpec(
        node_type="customdropout",
        display_name="Dropout",
        builder=_build_dropout,
        default_params={"rate": 0.5},
        description="Randomly zero-out units to prevent overfitting.",
        category="regularization",
    ),
    "custombatchnorm": LayerSpec(
        node_type="custombatchnorm",
        display_name="BatchNormalization",
        builder=_build_batchnorm,
        default_params={"momentum": 0.99, "epsilon": 0.001},
        description="Normalize activations of the previous layer at each batch.",
        category="regularization",
    ),
    "customlayernorm": LayerSpec(
        node_type="customlayernorm",
        display_name="LayerNormalization",
        builder=_build_layernorm,
        default_params={"epsilon": 0.001},
        description="Normalize across the features of each individual sample.",
        category="regularization",
    ),
    "customlstm": LayerSpec(
        node_type="customlstm",
        display_name="LSTM",
        builder=_build_lstm,
        default_params={"units": 64, "return_sequences": "false", "dropout": 0.0},
        description="Long Short-Term Memory recurrent layer.",
        category="recurrent",
    ),
    "customgru": LayerSpec(
        node_type="customgru",
        display_name="GRU",
        builder=_build_gru,
        default_params={"units": 64, "return_sequences": "false", "dropout": 0.0},
        description="Gated Recurrent Unit layer.",
        category="recurrent",
    ),
    "customsimplernn": LayerSpec(
        node_type="customsimplernn",
        display_name="SimpleRNN",
        builder=_build_simplernn,
        default_params={"units": 32, "return_sequences": "false"},
        description="Fully-connected RNN where the output is fed back as input.",
        category="recurrent",
    ),
    "customreshape": LayerSpec(
        node_type="customreshape",
        display_name="Reshape",
        builder=_build_reshape,
        default_params={"target_shape": "7,7,64"},
        description="Reshape the input to the given shape (comma-separated dims).",
        category="utility",
    ),
    "customembedding": LayerSpec(
        node_type="customembedding",
        display_name="Embedding",
        builder=_build_embedding,
        default_params={"input_dim": 1000, "output_dim": 64},
        description="Turn positive integers (word indices) into dense vectors.",
        category="utility",
    ),
}


def build_layer(node: dict, input_tensor) -> Any:
    """Dispatch a ReactFlow node to its Keras builder. Raises ValueError for unknown types."""
    node_type = node["type"]
    name = node["id"]
    params = node["data"]["params"]
    spec = LAYER_REGISTRY.get(node_type)
    if spec is None or spec.builder is None:
        raise ValueError(f"Unknown node type: {node_type!r}")
    logger.debug("Building layer type=%s name=%s", node_type, name)
    return spec.builder(params, input_tensor, name)


def get_layer_schema() -> list[dict]:
    """Return serialisable metadata for the frontend sidebar."""
    return [
        {
            "node_type": s.node_type,
            "display_name": s.display_name,
            "default_params": s.default_params,
            "description": s.description,
            "category": s.category,
        }
        for s in LAYER_REGISTRY.values()
    ]
