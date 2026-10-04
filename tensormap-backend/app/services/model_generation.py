"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
from collections import defaultdict

from app.layers.registry import LAYER_REGISTRY
from app.shared.errors import layer_shape_mismatch_message
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _node_params(node: dict) -> dict:
    """Return the node's ``data.params`` dict, or raise a descriptive ValueError.

    Malformed nodes that reach the legacy generator (e.g. because the IR
    translation failed first) previously crashed with a bare KeyError/TypeError
    on the missing structure. Raising ValueError with the node id lets the
    service layer surface the problem as a clean 400 response.
    """
    data = node.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("params"), dict):
        raise ValueError(
            f"Node {node.get('id', '<unknown>')} is missing its 'data.params' configuration "
            "(expected an object with the layer settings)."
        )
    return data["params"]


def _require_node_params(node: dict, *required: str) -> dict:
    """Return the node's params dict after ensuring every required key exists.

    Raises:
        ValueError: Naming the node id and every missing required field, so the
            client sees which parameter to fix instead of a raw KeyError.
    """
    params = _node_params(node)
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(
            f"Node {node.get('id', '<unknown>')} is missing required parameter(s) "
            f"{', '.join(repr(key) for key in missing)} in data.params."
        )
    return params


def _coerce_param(node: dict, key: str, value, cast=int):
    """Convert a raw node parameter with ``cast`` (``int`` or ``float``).

    Raises:
        ValueError: Naming the node id, the parameter and the value received,
            instead of the bare ``int()``/``float()`` error.
    """
    try:
        return cast(value)
    except (TypeError, ValueError) as e:
        expected = "a whole number" if cast is int else "a number"
        raise ValueError(
            f"Node {node.get('id', '<unknown>')}: parameter {key!r} must be {expected}, got {value!r}."
        ) from e


def _apply_layer(layer, input_tensor, node_id: str, source_ids):
    """Apply a built Keras layer to its input tensor.

    Keras rejects an incompatible input with a ValueError that only names the
    Keras layer and its ndim. Re-raise it as a message naming both canvas nodes,
    the incoming shape and what to change, keeping the original as the cause.
    """
    try:
        return layer(input_tensor)
    except ValueError as e:
        raise ValueError(layer_shape_mismatch_message(node_id, source_ids, input_tensor.shape, layer)) from e


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

    nodes_by_id = {node["id"]: node for node in model_params["nodes"]}

    # An edge may reference a source or target node that does not exist on the
    # canvas. Validate both endpoints up front (mirroring validate_ir_graph in
    # app/ir/schema.py) so a dangling edge is rejected with a message naming the
    # edge and the missing node, instead of a bare KeyError below or a generic
    # "not connected to any input layer" error later in the BFS.
    for edge in model_params["edges"]:
        source_id = edge["source"]
        target_id = edge["target"]
        if source_id not in nodes_by_id:
            raise ValueError(
                f"Edge {source_id} -> {target_id} references a source node that does not exist in the graph: "
                f"{source_id}. Remove this edge from the canvas."
            )
        if target_id not in nodes_by_id:
            raise ValueError(
                f"Edge {source_id} -> {target_id} references a target node that does not exist in the graph: "
                f"{target_id}. Remove this edge from the canvas."
            )

    # Build adjacency maps
    source_to_targets = defaultdict(list)
    target_to_sources = defaultdict(list)
    for edge in model_params["edges"]:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    input_nodes = [node for node in model_params["nodes"] if node["type"] == "custominput"]
    if not input_nodes:
        raise ValueError("Model has no input layer. Add an Input node to the canvas before training.")

    # BFS from input nodes to build Keras layers in topological order
    keras_tensors = {}
    visited = set()
    queue = []

    for node in input_nodes:
        input_params = _node_params(node)
        dims = [_coerce_param(node, f"dim-{i + 1}", input_params.get(f"dim-{i + 1}", 0) or 0) for i in range(3)]
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
            if len(source_tensors) > 1:
                input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
            else:
                input_tensor = source_tensors[0]

            node = nodes_by_id[target_id]
            keras_tensors[target_id] = _build_layer(node, input_tensor, all_sources)

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


def _build_layer(node: dict, input_tensor, source_ids=()):
    """Instantiate a single Keras layer from a ReactFlow node and apply it to the input tensor.

    ``source_ids`` are the ids of the nodes feeding ``input_tensor``; they are
    only used to name the upstream node in a shape mismatch error.

    LEGACY: This function is maintained for backward compatibility with the old
    ReactFlow format. New code should use TensorFlowGenerator._build_layer() instead.
    """
    # Lazy TF import
    import tensorflow as tf  # noqa: PLC0415

    params = _node_params(node)
    node_type = node["type"]
    name = node["id"]

    if node_type == "customdense":
        _require_node_params(node, "units", "activation")
        activation = params["activation"]
        layer = tf.keras.layers.Dense(
            units=_coerce_param(node, "units", params["units"]),
            activation="linear" if activation == "none" else activation,
            name=name,
        )

    elif node_type == "customflatten":
        layer = tf.keras.layers.Flatten(name=name)

    elif node_type == "custommaxpool":
        layer = tf.keras.layers.MaxPooling2D(
            pool_size=_coerce_param(node, "pool_size", params.get("pool_size", 2)),
            strides=_coerce_param(node, "stride", params.get("stride", 2)),
            padding=params.get("padding", "valid"),
            name=name,
        )

    elif node_type == "customglobalavgpool":
        layer = tf.keras.layers.GlobalAveragePooling2D(name=name)

    elif node_type == "customconv":
        _require_node_params(node, "filter", "kernelX", "kernelY", "strideX", "strideY", "padding", "activation")
        activation = params["activation"]
        layer = tf.keras.layers.Conv2D(
            filters=_coerce_param(node, "filter", params["filter"]),
            kernel_size=(
                _coerce_param(node, "kernelX", params["kernelX"]),
                _coerce_param(node, "kernelY", params["kernelY"]),
            ),
            strides=(
                _coerce_param(node, "strideX", params["strideX"]),
                _coerce_param(node, "strideY", params["strideY"]),
            ),
            padding=params["padding"],
            activation="linear" if activation == "none" else activation,
            name=name,
        )

    elif node_type == "customdropout":
        rate = _coerce_param(node, "rate", params.get("rate", 0.5), float)
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate!r}.")
        layer = tf.keras.layers.Dropout(rate=rate, name=name)

    else:
        raise ValueError(
            f"Unknown node type: {node_type} (node {name}). Supported layer types: {', '.join(sorted(LAYER_REGISTRY))}."
        )

    return _apply_layer(layer, input_tensor, name, source_ids)
