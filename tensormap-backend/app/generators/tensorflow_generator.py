"""TensorFlow/Keras model generator driven by the layer registry.

This module generates tf.keras.Model instances from IRGraph representations
using a registry-driven approach. TensorFlow is lazy-imported to avoid
import-time overhead during tests.

Architecture:
    IRGraph → BFS Traversal → Registry Lookup → Keras Layer Construction → Model

Key Design Decisions:
    1. Lazy TF Import: TensorFlow is imported inside methods, not at module level
    2. Registry-Driven: Uses LAYER_REGISTRY to map layer_type → Keras class
    3. BFS Traversal: Builds layers in topological order from input nodes
    4. Merge Handling: Concatenate and other merge layers handled specially
"""

from collections import defaultdict, deque
from typing import Any

from app.ir.schema import IRGraph, IRNode
from app.ir.translator import ir_to_keras_build_args
from app.layers.registry import LAYER_REGISTRY
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class TensorFlowGeneratorError(Exception):
    """Raised when model generation fails."""

    pass


class TensorFlowGenerator:
    """Generates a tf.keras.Model from an IRGraph using registry-driven layer construction.

    Example:
        >>> from app.ir.schema import IRGraph, IRNode, InputParams, DenseParams
        >>> graph = IRGraph(
        ...     nodes=[
        ...         IRNode(id="n1", node_params=InputParams(layer_type="input", shape=784)),
        ...         IRNode(id="n2", node_params=DenseParams(layer_type="dense", units=128, activation="relu")),
        ...     ],
        ...     edges=[IREdge(id="e1", source_id="n1", target_id="n2")]
        ... )
        >>> generator = TensorFlowGenerator()
        >>> model = generator.build_model(graph)
    """

    def build_model(self, graph: IRGraph):  # type: ignore
        """Generate a tf.keras.Model from an IRGraph.

        Uses BFS traversal starting from input nodes to build layers in
        topological order. Supports merge layers (Concatenate) and
        multi-input/multi-output architectures.

        Args:
            graph: The IRGraph to convert to a Keras model

        Returns:
            A compiled tf.keras.Model ready for training

        Raises:
            TensorFlowGeneratorError: If model generation fails (cycle, missing node, etc.)
        """
        # Lazy TF import to avoid startup penalty
        import tensorflow as tf  # noqa: PLC0415

        logger.debug("Building model from IRGraph with %d nodes, %d edges", len(graph.nodes), len(graph.edges))

        # Build adjacency maps for BFS traversal
        source_to_targets = defaultdict(list)
        target_to_sources = defaultdict(list)

        for edge in graph.edges:
            source_to_targets[edge.source_id].append(edge.target_id)
            target_to_sources[edge.target_id].append(edge.source_id)

        nodes_by_id = {node.id: node for node in graph.nodes}

        # Keras tensors for each node (populated during BFS)
        keras_tensors: dict[str, Any] = {}
        visited = set()
        queue = deque()

        # Start from input nodes - special handling since they return tensors directly
        input_nodes = [n for n in graph.nodes if n.node_params.layer_type == "input"]
        if not input_nodes:
            raise TensorFlowGeneratorError("Graph must contain at least one input node")

        for node in input_nodes:
            # Input nodes use tf.keras.Input which returns a tensor, not a layer
            build_args = ir_to_keras_build_args(node)
            build_args["name"] = node.id
            keras_tensors[node.id] = tf.keras.Input(**build_args)
            visited.add(node.id)
            queue.append(node.id)

        # BFS traversal to build all layers
        while queue:
            current_id = queue.popleft()

            # Process each outgoing edge
            for target_id in source_to_targets.get(current_id, []):
                if target_id in visited:
                    continue

                # Wait until all incoming edges are processed
                all_sources = target_to_sources.get(target_id, [])
                if not all(src in visited for src in all_sources):
                    continue

                visited.add(target_id)
                queue.append(target_id)

                node = nodes_by_id[target_id]

                # Gather input tensors
                source_tensors = [keras_tensors[src] for src in all_sources]

                # Handle merge layers (Concatenate)
                if node.node_params.layer_type == "concatenate":
                    if len(source_tensors) < 2:
                        raise TensorFlowGeneratorError(
                            f"Concatenate node {target_id} requires at least 2 inputs, got {len(source_tensors)}"
                        )
                    layer = self._build_layer(node)
                    keras_tensors[target_id] = layer(source_tensors)
                else:
                    # Single input or auto-concatenate
                    if len(source_tensors) > 1:
                        input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
                    else:
                        input_tensor = source_tensors[0]

                    layer = self._build_layer(node)
                    keras_tensors[target_id] = layer(input_tensor)

        # Identify inputs and outputs
        inputs = [keras_tensors[n.id] for n in input_nodes]

        # Output nodes are those with no outgoing edges
        output_node_ids = [nid for nid in nodes_by_id if nid not in source_to_targets]
        if not output_node_ids:
            raise TensorFlowGeneratorError("Graph has no output nodes (cycle or malformed graph)")

        outputs = [keras_tensors[oid] for oid in output_node_ids if oid in keras_tensors]

        if not outputs:
            raise TensorFlowGeneratorError("No outputs could be built (disconnected graph)")

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        logger.info("Model built successfully: %d layers, %d params", len(model.layers), model.count_params())

        return model

    def _build_layer(self, node: IRNode):  # type: ignore
        """Build a single Keras layer from an IRNode using the registry.

        This replaces the hardcoded elif chain with registry-driven construction.

        Args:
            node: The IRNode containing layer type and parameters

        Returns:
            A Keras layer instance (not yet applied to input tensor)

        Raises:
            TensorFlowGeneratorError: If layer type not found in registry or construction fails
        """
        # Lazy TF import

        layer_type = node.node_params.layer_type

        # Look up layer spec in registry
        if layer_type not in LAYER_REGISTRY:
            raise TensorFlowGeneratorError(f"Unknown layer type: {layer_type}")

        spec = LAYER_REGISTRY[layer_type]

        # Convert IRNode params to Keras constructor kwargs
        build_args = ir_to_keras_build_args(node)

        # Resolve Keras class from string path (e.g., "tf.keras.layers.Dense")
        keras_class = self._resolve_keras_class(spec.keras_class)

        # Special handling for activation="none" (use "linear" in Keras)
        if "activation" in build_args and build_args["activation"] == "none":
            build_args["activation"] = "linear"

        # Add name parameter for debugging/visualization
        build_args["name"] = node.id

        try:
            return keras_class(**build_args)
        except Exception as e:
            raise TensorFlowGeneratorError(f"Failed to build layer {layer_type} for node {node.id}: {e}") from e

    def _resolve_keras_class(self, class_path: str) -> type:
        """Resolve a Keras class from a string path like 'tf.keras.layers.Dense'.

        Args:
            class_path: Dot-separated path to the Keras class

        Returns:
            The resolved class object

        Raises:
            TensorFlowGeneratorError: If class cannot be resolved
        """
        # Lazy TF import
        import tensorflow as tf  # noqa: PLC0415

        parts = class_path.split(".")
        if parts[0] != "tf":
            raise TensorFlowGeneratorError(f"Invalid Keras class path: {class_path} (must start with 'tf.')")

        obj = tf
        for part in parts[1:]:
            try:
                obj = getattr(obj, part)
            except AttributeError as e:
                raise TensorFlowGeneratorError(f"Cannot resolve Keras class: {class_path}") from e

        return obj
