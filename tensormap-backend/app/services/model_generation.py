"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
import re
from collections import defaultdict

import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class ModelValidationError(ValueError):
    """User-facing validation error for graph/model issues."""


def _sanitize_tf_error(msg: str) -> str:
    """Strip stack trace and file-system path noise from TensorFlow/Keras errors."""
    # Remove common Python traceback file references.
    msg = re.sub(r'File\s+["\'][^"\']+\.py["\'](?:,\s*line\s*\d+)?', "", msg)
    # Remove stray POSIX/Windows python-file paths with optional line numbers.
    msg = re.sub(r'(?:[A-Za-z]:)?(?:[\\/][^\s,:"\']+)+\.py(?::\d+)?', "", msg)
    # Collapse whitespace/newlines introduced by replacements.
    msg = re.sub(r"\s+", " ", msg).strip(" :-")
    return msg or "TensorFlow reported an invalid model configuration"


def _validate_int_param(node_id: str, param_name: str, raw_value, allow_zero: bool = True) -> int:
    """Validate integer-like node parameters and raise a user-facing error if invalid."""
    if raw_value is None or raw_value == "":
        expected = "a non-negative integer" if allow_zero else "a positive integer"
        raise ModelValidationError(f"Missing value for '{param_name}' on node '{node_id}': expected {expected}.")

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ModelValidationError(
            f"Invalid value for '{param_name}' on node '{node_id}': expected an integer, got '{raw_value}'."
        ) from None

    if value < 0 or (not allow_zero and value == 0):
        expected = "a non-negative integer" if allow_zero else "a positive integer"
        raise ModelValidationError(
            f"Invalid value for '{param_name}' on node '{node_id}': expected {expected}, got '{value}'."
        )

    return value


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure.

    Builds the model programmatically using the Keras API, then serializes
    it via ``model.to_json()`` so the output always matches the installed
    Keras version's expected format.
    """
    nodes = model_params.get("nodes", [])
    edges = model_params.get("edges", [])
    logger.debug("Generating model from %d nodes, %d edges", len(nodes), len(edges))

    input_nodes = [node for node in nodes if node.get("type") == "custominput"]
    if not input_nodes:
        raise ModelValidationError("No Input node found. Add at least one Input node to start the model.")

    # Build node lookup map
    nodes_by_id = {node["id"]: node for node in nodes}
    # Pre-validate that all edges reference existing node IDs
    for edge in edges:
        source_id = edge.get("source")
        target_id = edge.get("target")
        if source_id not in nodes_by_id:
            raise ModelValidationError(
                f"Edge references unknown source node id '{source_id}'. Every edge must connect existing nodes."
            )
        if target_id not in nodes_by_id:
            raise ModelValidationError(
                f"Edge from '{source_id}' references unknown target node id '{target_id}'. "
                "Every edge must connect existing nodes."
            )

    # Build adjacency maps
    source_to_targets = defaultdict(list)
    target_to_sources = defaultdict(list)
    for edge in edges:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    # BFS from input nodes to build Keras layers in topological order
    keras_tensors = {}
    visited = set()
    queue = []

    for node in input_nodes:
        params = node.get("data", {}).get("params", {})
        dims = []
        for i in range(3):
            param_name = f"dim-{i + 1}"
            dim = _validate_int_param(node["id"], param_name, params.get(param_name, 0))
            if dim != 0:
                dims.append(dim)

        if not dims:
            raise ModelValidationError(
                f"Input node '{node['id']}' has no valid dimensions. Provide at least one non-zero dimension."
            )

        try:
            keras_tensors[node["id"]] = tf.keras.Input(shape=dims, name=node["id"])
        except Exception as e:
            raise ModelValidationError(
                f"Invalid Input configuration on node '{node['id']}': {_sanitize_tf_error(str(e))}"
            ) from None

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

    unvisited_ids = [node_id for node_id in nodes_by_id if node_id not in visited]
    if unvisited_ids:
        joined = ", ".join(unvisited_ids)
        raise ModelValidationError(
            f"Disconnected graph detected: {len(unvisited_ids)} unreachable node(s): {joined}. "
            "Ensure all nodes are connected to an Input path."
        )

    # Identify input and output tensors
    inputs = [keras_tensors[n["id"]] for n in input_nodes]
    output_ids = [node_id for node_id in nodes_by_id if node_id not in source_to_targets]
    if not output_ids:
        raise ModelValidationError("No output node found. Add at least one node with no outgoing edges.")
    outputs = [keras_tensors[oid] for oid in output_ids]

    try:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return json.loads(model.to_json())
    except Exception as e:
        sanitized = _sanitize_tf_error(str(e))
        hint = ""
        if any(word in sanitized.lower() for word in ["shape", "incompatible", "rank"]):
            hint = " If this is a dimension mismatch, add a Flatten layer before Dense or adjust layer shapes."
        raise ModelValidationError(f"Model validation failed: {sanitized}.{hint}".rstrip()) from None


def _build_layer(node: dict, input_tensor):
    """Instantiate a single Keras layer from a ReactFlow node and apply it to the input tensor."""
    params = node.get("data", {}).get("params", {})
    node_type = node["type"]
    name = node["id"]

    if node_type == "customdense":
        units = _validate_int_param(name, "units", params.get("units"), allow_zero=False)
        activation = params.get("activation", "linear")
        try:
            return tf.keras.layers.Dense(
                units=units,
                activation="linear" if activation == "none" else activation,
                name=name,
            )(input_tensor)
        except Exception as e:
            raise ModelValidationError(
                f"Failed to apply Dense on node '{name}': {_sanitize_tf_error(str(e))}"
            ) from None

    elif node_type == "customflatten":
        return tf.keras.layers.Flatten(name=name)(input_tensor)

    elif node_type == "customdropout":
        return tf.keras.layers.Dropout(
            rate=float(params.get("rate", 0.5)),
            name=name,
        )(input_tensor)
    elif node_type == "custommaxpool":
        return tf.keras.layers.MaxPooling2D(
            pool_size=int(params.get("pool_size", 2)),
            strides=int(params.get("stride", 2)),
            padding=params.get("padding", "valid"),
            name=name,
        )(input_tensor)
    elif node_type == "customconv":
        filters = _validate_int_param(name, "filter", params.get("filter"), allow_zero=False)
        kernel_x = _validate_int_param(name, "kernelX", params.get("kernelX"), allow_zero=False)
        kernel_y = _validate_int_param(name, "kernelY", params.get("kernelY"), allow_zero=False)
        stride_x = _validate_int_param(name, "strideX", params.get("strideX"), allow_zero=False)
        stride_y = _validate_int_param(name, "strideY", params.get("strideY"), allow_zero=False)
        padding = params.get("padding", "valid")
        if padding not in {"valid", "same"}:
            raise ModelValidationError(
                f"Invalid value for 'padding' on node '{name}': expected 'valid' or 'same', got '{padding}'."
            )
        activation = params.get("activation", "linear")

        try:
            return tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_x, kernel_y),
                strides=(stride_x, stride_y),
                padding=padding,
                activation="linear" if activation == "none" else activation,
                name=name,
            )(input_tensor)
        except Exception as e:
            raise ModelValidationError(
                f"Failed to apply Conv2D on node '{name}': {_sanitize_tf_error(str(e))}"
            ) from None

    else:
        raise ModelValidationError(f"Unknown node type: {node_type}")
