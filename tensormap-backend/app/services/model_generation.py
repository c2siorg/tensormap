"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
from collections import defaultdict

import tensorflow as tf

from app.layers.registry import LAYER_REGISTRY, build_layer
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure."""
    nodes = model_params["nodes"]
    edges = model_params["edges"]
    logger.debug("Generating model from %d nodes, %d edges", len(nodes), len(edges))

    source_to_targets: dict[str, list[str]] = defaultdict(list)
    target_to_sources: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    nodes_by_id = {node["id"]: node for node in nodes}
    keras_tensors: dict = {}
    visited: set[str] = set()
    queue: list[str] = []

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
            all_sources = target_to_sources.get(target_id, [])
            if not all(src in visited for src in all_sources):
                continue
            visited.add(target_id)
            queue.append(target_id)
            source_tensors = [keras_tensors[src] for src in all_sources]
            input_tensor = (
                tf.keras.layers.Concatenate(axis=-1)(source_tensors) if len(source_tensors) > 1 else source_tensors[0]
            )
            node = nodes_by_id[target_id]
            keras_tensors[target_id] = build_layer(node, input_tensor)

    inputs = [keras_tensors[n["id"]] for n in nodes if n["type"] == "custominput"]
    output_ids = [n["id"] for n in nodes if n["id"] not in source_to_targets]
    outputs = [keras_tensors[oid] for oid in output_ids]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return json.loads(model.to_json())


def get_supported_layer_types() -> list[str]:
    """Return all registered node type keys."""
    return list(LAYER_REGISTRY.keys())
