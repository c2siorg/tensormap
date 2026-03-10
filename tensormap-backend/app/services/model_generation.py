"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
import re
from collections import defaultdict

import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class ModelValidationError(Exception):
    """User-friendly error raised when model graph validation fails."""


_SUPPORTED_LAYER_TYPES = {
    "custominput": "Input",
    "customdense": "Dense",
    "customconv": "Conv2D",
    "customflatten": "Flatten",
}


def _sanitize_tf_error(msg: str) -> str:
    """Strip internal file paths and stack-trace noise from a TensorFlow error message."""
    msg = re.sub(r"(?:File |/)[\w/\\.\-]+\.py[:\d]*", "", msg)
    msg = re.sub(r"\s{2,}", " ", msg).strip()
    return msg


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure.

    Builds the model programmatically using the Keras API, then serializes
    it via ``model.to_json()`` so the output always matches the installed
    Keras version's expected format.

    Raises ``ModelValidationError`` with a user-friendly message when the
    graph is invalid (missing Input, disconnected nodes, bad parameters, etc.).
    """
    logger.debug("Generating model from %d nodes, %d edges", len(model_params["nodes"]), len(model_params["edges"]))

    nodes = model_params["nodes"]
    edges = model_params["edges"]

    # --- Pre-validation: Input nodes exist ---
    input_nodes = [n for n in nodes if n["type"] == "custominput"]
    if not input_nodes:
        raise ModelValidationError(
            "No Input layer found. Every model needs at least one Input node as the starting point of the network."
        )

    # --- Pre-validation: Unknown node types ---
    for node in nodes:
        if node["type"] not in _SUPPORTED_LAYER_TYPES:
            supported = ", ".join(sorted(_SUPPORTED_LAYER_TYPES.values()))
            raise ModelValidationError(
                f"Unknown layer type '{node['type']}' on node '{node['id']}'. Supported types: {supported}."
            )

    # Build adjacency maps
    source_to_targets = defaultdict(list)
    target_to_sources = defaultdict(list)
    for edge in edges:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    nodes_by_id = {node["id"]: node for node in nodes}

    # BFS from input nodes to build Keras layers in topological order
    keras_tensors = {}
    visited = set()
    queue = []

    for node in input_nodes:
        dims = [int(node["data"]["params"].get(f"dim-{i + 1}", 0) or 0) for i in range(3)]
        dims = [d for d in dims if d != 0]
        keras_tensors[node["id"]] = tf.keras.Input(shape=dims, name=node["id"])
        visited.add(node["id"])
        queue.append(node["id"])

    while queue:
        current_id = queue.pop(0)
        for target_id in source_to_targets.get(current_id, []):
            if target_id in visited:
                continue
            # Check if all sources of this target have been visited
            all_sources = target_to_sources.get(target_id, [])
            if not all(src in visited for src in all_sources):
                continue

            visited.add(target_id)
            queue.append(target_id)

            # Collect input tensors for this node
            source_tensors = [keras_tensors[src] for src in all_sources]
            if len(source_tensors) > 1:
                input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
            else:
                input_tensor = source_tensors[0]

            node = nodes_by_id[target_id]
            keras_tensors[target_id] = _build_layer(node, input_tensor)

    # --- Post-BFS validation: Disconnected graph ---
    all_node_ids = {n["id"] for n in nodes}
    disconnected = all_node_ids - visited
    if disconnected:
        names = ", ".join(sorted(disconnected))
        raise ModelValidationError(
            f"Disconnected graph: {len(disconnected)} node(s) are not connected "
            f"to the Input layer. Ensure every layer is reachable via edges from "
            f"an Input node. Disconnected nodes: {names}."
        )

    # Identify input and output tensors
    inputs = [keras_tensors[n["id"]] for n in nodes if n["type"] == "custominput"]
    output_ids = [n["id"] for n in nodes if n["id"] not in source_to_targets]
    outputs = [keras_tensors[oid] for oid in output_ids]

    try:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    except Exception as e:
        msg = str(e)
        # Detect shape incompatibilities and suggest fixes
        if "shape" in msg.lower() or "incompatible" in msg.lower():
            raise ModelValidationError(
                f"Layer shape mismatch: the connected layers have incompatible "
                f"shapes. If you are connecting a Conv2D layer to a Dense layer, "
                f"add a Flatten layer in between to convert feature maps to a 1D "
                f"vector. Detail: {_sanitize_tf_error(msg)}"
            ) from None
        raise ModelValidationError(f"Model construction failed: {_sanitize_tf_error(msg)}") from None

    return json.loads(model.to_json())


def _validate_int_param(node_id: str, param_name: str, value) -> int:
    """Convert a parameter to int, raising ModelValidationError on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ModelValidationError(
            f"Invalid value for '{param_name}' on node '{node_id}': expected a numeric value, got '{value}'."
        ) from None


def _build_layer(node: dict, input_tensor):
    """Instantiate a single Keras layer from a ReactFlow node and apply it to the input tensor."""
    params = node["data"]["params"]
    node_type = node["type"]
    name = node["id"]

    if node_type == "customdense":
        units = _validate_int_param(name, "units", params.get("units"))
        activation = params.get("activation", "linear")
        return tf.keras.layers.Dense(
            units=units,
            activation="linear" if activation == "none" else activation,
            name=name,
        )(input_tensor)

    elif node_type == "customflatten":
        return tf.keras.layers.Flatten(name=name)(input_tensor)

    elif node_type == "customconv":
        filters = _validate_int_param(name, "filter", params.get("filter"))
        kernel_x = _validate_int_param(name, "kernelX", params.get("kernelX"))
        kernel_y = _validate_int_param(name, "kernelY", params.get("kernelY"))
        stride_x = _validate_int_param(name, "strideX", params.get("strideX"))
        stride_y = _validate_int_param(name, "strideY", params.get("strideY"))
        activation = params.get("activation", "linear")
        try:
            return tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_x, kernel_y),
                strides=(stride_x, stride_y),
                padding=params.get("padding", "valid"),
                activation="linear" if activation == "none" else activation,
                name=name,
            )(input_tensor)
        except Exception as e:
            raise ModelValidationError(
                f"Failed to apply Conv2D on node '{name}': {_sanitize_tf_error(str(e))}"
            ) from None

    else:
        supported = ", ".join(sorted(_SUPPORTED_LAYER_TYPES.values()))
        raise ModelValidationError(f"Unknown layer type '{node_type}' on node '{name}'. Supported types: {supported}.")
