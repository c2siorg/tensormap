"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
from collections import defaultdict

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure.

    This is the legacy entry point that maintains backward compatibility with
    the old ReactFlow format. New code should use build_model_from_ir() instead.

    Builds the model programmatically using the Keras API, then serializes
    it via ``model.to_json()`` so the output always matches the installed
    Keras version's expected format.
    """
    # Lazy TF import to avoid startup penalty
    import tensorflow as tf  # noqa: PLC0415

    logger.debug(
        "Generating model from %d nodes, %d edges",
        len(model_params["nodes"]),
        len(model_params["edges"]),
    )

    # Build adjacency maps
    source_to_targets = defaultdict(list)
    target_to_sources = defaultdict(list)
    for edge in model_params["edges"]:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    nodes_by_id = {node["id"]: node for node in model_params["nodes"]}

    input_nodes = [node for node in model_params["nodes"] if node["type"] == "custominput"]
    if not input_nodes:
        raise ValueError("Model has no input layer. Add an Input node to the canvas before training.")

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

            all_sources = target_to_sources.get(target_id, [])
            if not all(src in visited for src in all_sources):
                continue

            visited.add(target_id)
            queue.append(target_id)

            # An edge may reference a target node that does not exist on the
            # canvas. Reject it with a clear message instead of crashing with a
            # bare KeyError on the lookup below.
            if target_id not in nodes_by_id:
                raise ValueError(
                    f"Edge {current_id} -> {target_id} references a node that does not exist in the graph. "
                    "Remove this edge from the canvas."
                )

            source_tensors = [keras_tensors[src] for src in all_sources]
            if len(source_tensors) > 1:
                input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
            else:
                input_tensor = source_tensors[0]

            node = nodes_by_id[target_id]
            keras_tensors[target_id] = _build_layer(node, input_tensor)

    # Every node must be reachable from an input layer. Anything left unvisited is
    # either an orphan the user forgot to wire up or part of a cycle -- the BFS never
    # built a tensor for it, so the output lookup below would fail with a bare
    # KeyError. Fail early and name the offending nodes so the user can find them.
    unreachable = sorted(node_id for node_id in nodes_by_id if node_id not in visited)
    if unreachable:
        raise ValueError(
            f"Model contains layers that are not connected to any input layer: {', '.join(unreachable)}. "
            "Connect them to the graph or remove them from the canvas."
        )

    inputs = [keras_tensors[node["id"]] for node in input_nodes]

    # A node with no outgoing edges is a model output -- except for input nodes, which
    # are dangling rather than terminal. Treating them as outputs silently produced a
    # malformed model whose input was also its output.
    output_ids = [
        node["id"]
        for node in model_params["nodes"]
        if node["id"] not in source_to_targets and node["type"] != "custominput"
    ]
    if not output_ids:
        raise ValueError(
            "Model has no output layer. Connect at least one layer downstream of an input node before training."
        )

    # Input nodes are seeded into `visited` before the BFS runs, so the reachability
    # check above cannot catch a dangling one. Report it here rather than letting
    # Keras fail later with an opaque "`inputs` not connected to `outputs`".
    dangling_inputs = sorted(node["id"] for node in input_nodes if node["id"] not in source_to_targets)
    if dangling_inputs:
        raise ValueError(
            f"Input layers are not connected to any other layer: {', '.join(dangling_inputs)}. "
            "Connect them to the graph or remove them from the canvas."
        )

    outputs = [keras_tensors[oid] for oid in output_ids]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return json.loads(model.to_json())


def build_model_from_ir(graph: "IRGraph") -> dict:  # type: ignore  # noqa: F821
    """Build a Keras model from an IRGraph and return its JSON representation.

    This is the new, registry-driven entry point that replaces the legacy
    model_generation() function for IRGraph-based workflows.

    Args:
        graph: The validated IRGraph to convert

    Returns:
        Keras model JSON dictionary ready for serialization

    Example:
        >>> from app.ir.schema import IRGraph
        >>> from app.ir.translator import reactflow_to_ir
        >>> graph = reactflow_to_ir(canvas_json)
        >>> model_json = build_model_from_ir(graph)
    """
    from app.generators.tensorflow_generator import TensorFlowGenerator

    generator = TensorFlowGenerator()
    model = generator.build_model(graph)

    # Lazy TF import

    return json.loads(model.to_json())


def _build_layer(node: dict, input_tensor):
    """Instantiate a single Keras layer from a ReactFlow node and apply it to the input tensor.

    LEGACY: This function is maintained for backward compatibility with the old
    ReactFlow format. New code should use TensorFlowGenerator._build_layer() instead.
    """
    # Lazy TF import
    import tensorflow as tf  # noqa: PLC0415

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

    elif node_type == "custommaxpool":
        return tf.keras.layers.MaxPooling2D(
            pool_size=int(params.get("pool_size", 2)),
            strides=int(params.get("stride", 2)),
            padding=params.get("padding", "valid"),
            name=name,
        )(input_tensor)

    elif node_type == "customglobalavgpool":
        return tf.keras.layers.GlobalAveragePooling2D(name=name)(input_tensor)

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

    elif node_type == "customdropout":
        rate = float(params.get("rate", 0.5))
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate!r}.")
        return tf.keras.layers.Dropout(rate=rate, name=name)(input_tensor)

    else:
        raise ValueError(f"Unknown node type: {node_type}")
