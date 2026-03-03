"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
from collections import defaultdict

import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)

# Node types that model_generation() knows how to build.
SUPPORTED_NODE_TYPES = {"custominput", "customdense", "customflatten", "customconv"}


class ModelValidationError(ValueError):
    """Raised when the ReactFlow graph fails pre-build validation.

    Carries a ``user_message`` that is safe to return directly to the client
    (no internal paths, stack traces, or TensorFlow class names).
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


def _validate_graph(nodes: list[dict], edges: list[dict]) -> None:
    """Run pre-build checks on the ReactFlow graph and raise ModelValidationError on failure.

    Checks performed (in order):
    1. At least one ``custominput`` node is present.
    2. Every node type is in SUPPORTED_NODE_TYPES.
    3. All layer parameters that must be numeric are valid integers.
    4. Every non-input node is reachable from an input node (disconnected graph).
    """
    # ------------------------------------------------------------------
    # 1.  Must have at least one Input layer
    # ------------------------------------------------------------------
    input_node_ids = {n["id"] for n in nodes if n["type"] == "custominput"}
    if not input_node_ids:
        raise ModelValidationError(
            "No Input layer found. Add an Input node to your canvas and connect it "
            "to the rest of the network before validating."
        )

    # ------------------------------------------------------------------
    # 2. Unknown node types
    # ------------------------------------------------------------------
    unknown = [n for n in nodes if n["type"] not in SUPPORTED_NODE_TYPES]
    if unknown:
        labels = ", ".join(f'"{n["type"]}"' for n in unknown)
        supported = ", ".join(sorted(SUPPORTED_NODE_TYPES - {"custominput"}))
        raise ModelValidationError(
            f"Unknown layer type(s): {labels}. "
            f"Supported types are: custominput, {supported}."
        )

    # ------------------------------------------------------------------
    # 3. Parameter type validation
    # ------------------------------------------------------------------
    for node in nodes:
        node_id = node["id"]
        params = node["data"].get("params", {})
        node_type = node["type"]

        int_fields: dict[str, str] = {}
        if node_type == "customdense":
            int_fields = {"units": "Units"}
        elif node_type == "customconv":
            int_fields = {
                "filter": "Filters",
                "kernelX": "Kernel width",
                "kernelY": "Kernel height",
                "strideX": "Stride width",
                "strideY": "Stride height",
            }
        elif node_type == "custominput":
            for i in range(1, 4):
                key = f"dim-{i}"
                val = params.get(key)
                if val is not None and val != "" and val != 0:
                    try:
                        int(val)
                    except (TypeError, ValueError):
                        raise ModelValidationError(
                            f'Node "{node_id}": dimension "{key}" must be a positive integer, '
                            f'got "{val}". Fix the Input layer parameters and try again.'
                        ) from None

        for field, label in int_fields.items():
            val = params.get(field)
            try:
                int(val)
            except (TypeError, ValueError):
                raise ModelValidationError(
                    f'Node "{node_id}": "{label}" must be a positive integer, '
                    f'got "{val}". Fix the parameter and try again.'
                ) from None

    # ------------------------------------------------------------------
    # 4. Disconnected graph — BFS from all input nodes
    # ------------------------------------------------------------------
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["source"]].append(edge["target"])
        adjacency[edge["target"]].append(edge["source"])

    reachable: set[str] = set()
    queue = list(input_node_ids)
    reachable.update(input_node_ids)

    while queue:
        current = queue.pop(0)
        for neighbour in adjacency.get(current, []):
            if neighbour not in reachable:
                reachable.add(neighbour)
                queue.append(neighbour)

    all_ids = {n["id"] for n in nodes}
    disconnected = all_ids - reachable
    if disconnected:
        count = len(disconnected)
        raise ModelValidationError(
            f"Disconnected graph: {count} node(s) are not connected to the Input layer "
            f"({', '.join(sorted(disconnected))}). Connect every node to the network "
            "before validating."
        )


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure.

    Builds the model programmatically using the Keras API, then serializes
    it via ``model.to_json()`` so the output always matches the installed
    Keras version's expected format.

    Raises:
        ModelValidationError: Structured, user-friendly error for any graph or
            parameter problem detected *before* or *during* the Keras build.
    """
    nodes: list[dict] = model_params["nodes"]
    edges: list[dict] = model_params["edges"]

    logger.debug("Generating model from %d nodes, %d edges", len(nodes), len(edges))

    # -- Pre-build validation (fast-path, no TF involved) ----------------
    _validate_graph(nodes, edges)

    # Build adjacency maps
    source_to_targets: dict[str, list[str]] = defaultdict(list)
    target_to_sources: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    nodes_by_id = {node["id"]: node for node in nodes}

    # BFS from input nodes to build Keras layers in topological order
    keras_tensors: dict[str, object] = {}
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
            if len(source_tensors) > 1:
                input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
            else:
                input_tensor = source_tensors[0]

            node = nodes_by_id[target_id]
            try:
                keras_tensors[target_id] = _build_layer(node, input_tensor)
            except ModelValidationError:
                raise
            except (ValueError, TypeError) as exc:
                raise ModelValidationError(_wrap_build_error(node, exc)) from exc
            except Exception as exc:  # noqa: BLE001
                raise ModelValidationError(_wrap_build_error(node, exc)) from exc

    # Identify input and output tensors
    inputs = [keras_tensors[n["id"]] for n in nodes if n["type"] == "custominput"]
    output_ids = [n["id"] for n in nodes if n["id"] not in source_to_targets]
    outputs = [keras_tensors[oid] for oid in output_ids]

    try:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    except Exception as exc:  # noqa: BLE001
        raise ModelValidationError(_wrap_model_build_error(exc)) from exc

    return json.loads(model.to_json())


def _wrap_build_error(node: dict, exc: Exception) -> str:
    """Convert a low-level exception raised inside _build_layer into a user message."""
    node_id = node["id"]
    node_type = node["type"]
    err = str(exc)

    # Dense applied to a multi-dimensional tensor (e.g., Conv2D output)
    err_lower = err.lower()
    if node_type == "customdense" and (
        "rank" in err_lower or "dimensions" in err_lower or "shape" in err_lower
    ):
        return (
            f'Node "{node_id}" (Dense): received a multi-dimensional input. '
            "Dense layers require 1-D input — add a Flatten layer between the "
            "previous layer and this Dense layer."
        )

    # Conv2D applied to 1-D input
    if node_type == "customconv" and ("2d" in err.lower() or "rank" in err.lower()):
        return (
            f'Node "{node_id}" (Conv2D): Conv2D requires a 3-D input (height, width, channels). '
            "Check that the Input layer has three dimensions (e.g., 28, 28, 1)."
        )

    # Generic parameter error
    return (
        f'Node "{node_id}" ({node_type}): invalid configuration — {err}. '
        "Review the node parameters and try again."
    )


def _wrap_model_build_error(exc: Exception) -> str:
    """Convert a tf.keras.Model() exception into a user-friendly message."""
    err = str(exc)

    if "graph is disconnected" in err.lower():
        return (
            "The model graph is disconnected. Ensure every layer is reachable from "
            "an Input node and that all branches are connected."
        )

    if "incompatible" in err.lower() or "shape" in err.lower():
        return (
            "Shape mismatch detected while assembling the model. "
            "If you are connecting a Conv2D layer to a Dense layer, add a Flatten "
            "layer in between. Check that all layer shapes are compatible."
        )

    if "output" in err.lower() and ("same" in err.lower() or "input" in err.lower()):
        return (
            "The model output cannot be the same tensor as the input. "
            "Add at least one layer between the Input and the output."
        )

    # Fallback — strip any file path fragments before returning
    safe_err = err.split("\n")[0][:200]
    return f"Failed to build the model: {safe_err}. Review your network design and try again."


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
        raise ModelValidationError(
            f'Unknown layer type "{node_type}" on node "{name}". '
            f"Supported types are: {', '.join(sorted(SUPPORTED_NODE_TYPES - {'custominput'}))}."
        )
