"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
from collections import defaultdict

import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure.

    Builds the model programmatically using the Keras API, then serializes
    it via ``model.to_json()`` so the output always matches the installed
    Keras version's expected format.
    """
    logger.debug("Generating model from %d nodes, %d edges", len(model_params["nodes"]), len(model_params["edges"]))

    nodes = model_params.get("nodes", [])
    edges = model_params.get("edges", [])
    if not nodes:
        raise ValueError("Graph must include at least one node")

    node_ids = [node["id"] for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Duplicate node IDs are not allowed")

    input_ids = [node["id"] for node in nodes if node["type"] == "custominput"]
    if not input_ids:
        raise ValueError("Graph must include at least one input node")

    nodes_by_id = {node["id"]: node for node in nodes}

    # Build adjacency maps
    source_to_targets = defaultdict(list)
    target_to_sources = defaultdict(list)
    for edge in edges:
        if edge["source"] not in nodes_by_id or edge["target"] not in nodes_by_id:
            raise ValueError(f"Edge references unknown node(s): source={edge['source']}, target={edge['target']}")
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    for input_id in input_ids:
        if target_to_sources.get(input_id):
            raise ValueError(f"Input node '{input_id}' cannot have incoming edges")
        if not source_to_targets.get(input_id):
            raise ValueError(f"Input node '{input_id}' must connect to at least one layer")

    for node in nodes:
        if node["type"] != "custominput" and not target_to_sources.get(node["id"]):
            raise ValueError(f"Node '{node['id']}' has no incoming edges")

    # BFS from input nodes to build Keras layers in topological order
    keras_tensors = {}
    visited = set()
    queue = []

    for node in nodes:
        if node["type"] == "custominput":
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
            try:
                keras_tensors[target_id] = _build_layer(node, input_tensor)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Failed to build node '{target_id}': {e}") from e

    unresolved_nodes = [node_id for node_id in node_ids if node_id not in visited]
    if unresolved_nodes:
        raise ValueError(f"Graph contains disconnected or cyclic nodes: {sorted(unresolved_nodes)}")

    if all(nodes_by_id[node_id]["type"] == "custominput" for node_id in visited):
        raise ValueError("Graph must include at least one non-input layer connected to an input")

    # Identify input and output tensors
    inputs = [keras_tensors[n["id"]] for n in nodes if n["type"] == "custominput"]
    output_ids = [n["id"] for n in nodes if n["id"] not in source_to_targets]
    outputs = [keras_tensors[oid] for oid in output_ids]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return json.loads(model.to_json())


def _build_layer(node: dict, input_tensor):
    """Instantiate a single Keras layer from a ReactFlow node and apply it to the input tensor."""
    params = node["data"]["params"]
    node_type = node["type"]
    name = node["id"]

    if node_type == "customdense":
        activation = params["activation"]
        return tf.keras.layers.Dense(
            units=int(params["units"]),
            activation="linear" if activation == "none" else activation,
            name=name,
        )(input_tensor)

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
        activation = params["activation"]
        return tf.keras.layers.Conv2D(
            filters=int(params["filter"]),
            kernel_size=(int(params["kernelX"]), int(params["kernelY"])),
            strides=(int(params["strideX"]), int(params["strideY"])),
            padding=params["padding"],
            activation="linear" if activation == "none" else activation,
            name=name,
        )(input_tensor)

    else:
        raise ValueError(f"Unknown node type: {node_type}")
