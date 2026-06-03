"""
Tests for the layer registry and IR schema.

These tests verify:
1. Registry completeness and correctness
2. Parameter specification constraints
3. IR schema validation and round-trip serialization
4. Graph validation rules (DAG, input nodes, merge layers)
"""

import json

import pytest
from pydantic import ValidationError

from app.ir.schema import (
    ConcatenateParams,
    DenseParams,
    InputParams,
    IREdge,
    IRGraph,
    IRNode,
    IRValidationError,
    validate_ir_graph,
)
from app.layers.registry import (
    LAYER_REGISTRY,
    ParamType,
    get_layer_spec,
    serialize_registry,
)


class TestLayerRegistry:
    """Tests for the layer registry structure and specifications."""

    def test_all_15_layers_present(self) -> None:
        """Verify that all 15 expected layer types are present in the registry."""
        expected_layers = {
            "input",
            "dense",
            "flatten",
            "conv2d",
            "maxpool2d",
            "avgpool2d",
            "globalavgpool2d",
            "lstm",
            "gru",
            "simplernn",
            "embedding",
            "dropout",
            "batchnorm",
            "reshape",
            "concatenate",
        }
        assert set(LAYER_REGISTRY.keys()) == expected_layers, "Registry must contain exactly 15 layer types"

    def test_required_params_have_no_default(self) -> None:
        """Verify that required parameters have required=True and default=None."""
        for layer_key, layer_spec in LAYER_REGISTRY.items():
            for param_name, param_spec in layer_spec.params.items():
                if param_spec.required:
                    assert param_spec.default is None, (
                        f"Layer '{layer_key}', param '{param_name}': "
                        f"required params must have default=None, got {param_spec.default}"
                    )

    def test_enum_params_have_values_list(self) -> None:
        """Verify that all ENUM type parameters have a non-empty values list."""
        for layer_key, layer_spec in LAYER_REGISTRY.items():
            for param_name, param_spec in layer_spec.params.items():
                if param_spec.type == ParamType.ENUM:
                    assert param_spec.values is not None, (
                        f"Layer '{layer_key}', param '{param_name}': ENUM params must have values list"
                    )
                    assert len(param_spec.values) > 0, (
                        f"Layer '{layer_key}', param '{param_name}': ENUM values list must not be empty"
                    )

    def test_float_params_have_min_max(self) -> None:
        """Verify that FLOAT parameters have min and max constraints set."""
        for layer_key, layer_spec in LAYER_REGISTRY.items():
            for param_name, param_spec in layer_spec.params.items():
                if param_spec.type == ParamType.FLOAT:
                    assert param_spec.min is not None, (
                        f"Layer '{layer_key}', param '{param_name}': FLOAT params must have min constraint"
                    )
                    assert param_spec.max is not None, (
                        f"Layer '{layer_key}', param '{param_name}': FLOAT params must have max constraint"
                    )

    def test_concatenate_is_merge_only_layer(self) -> None:
        """Verify that only the 'concatenate' layer has merge=True."""
        merge_layers = [key for key, spec in LAYER_REGISTRY.items() if spec.merge]
        assert merge_layers == ["concatenate"], "Only 'concatenate' should have merge=True"

    def test_serialize_registry_is_json_serializable(self) -> None:
        """Verify that serialize_registry() produces valid JSON."""
        serialized = serialize_registry()
        assert isinstance(serialized, list), "Serialized registry must be a list"
        assert len(serialized) == 15, "Serialized registry must contain 15 layers"

        # Verify it can be JSON-encoded
        json_str = json.dumps(serialized)
        assert json_str is not None, "Serialized registry must be JSON-serializable"

        # Verify structure of first entry
        first_layer = serialized[0]
        required_keys = {"type_key", "display_name", "category", "keras_class", "params", "merge", "description"}
        assert set(first_layer.keys()) == required_keys, "Each layer must have all required keys"

    def test_get_layer_spec_success(self) -> None:
        """Verify that get_layer_spec() returns correct specs for valid keys."""
        dense_spec = get_layer_spec("dense")
        assert dense_spec.type_key == "dense"
        assert dense_spec.display_name == "Dense"
        assert "units" in dense_spec.params

    def test_get_layer_spec_unknown_key(self) -> None:
        """Verify that get_layer_spec() raises KeyError for unknown layer types."""
        with pytest.raises(KeyError, match="Unknown layer type: nonexistent"):
            get_layer_spec("nonexistent")


class TestIRSchema:
    """Tests for the IR schema Pydantic models and validation."""

    def test_ir_round_trip_dense(self) -> None:
        """Test serialization and deserialization of a Dense layer node."""
        # Create a Dense layer node
        dense_params = DenseParams(layer_type="dense", units=64, activation="relu")
        node = IRNode(
            id="node-1",
            node_params=dense_params,
            position={"x": 100.0, "y": 200.0},
            input_node_ids=[],
        )

        # Serialize to dict
        node_dict = node.model_dump()

        # Deserialize back
        node_restored = IRNode.model_validate(node_dict)

        # Verify equality
        assert node_restored.id == node.id
        assert node_restored.node_params.layer_type == "dense"
        assert node_restored.node_params.units == 64  # type: ignore
        assert node_restored.node_params.activation == "relu"  # type: ignore
        assert node_restored.position == {"x": 100.0, "y": 200.0}

    def test_ir_rejects_unknown_layer_type(self) -> None:
        """Verify that Pydantic raises ValidationError for unknown layer_type."""
        with pytest.raises(ValidationError):
            IRNode(
                id="node-1",
                node_params={"layer_type": "unknown_layer", "some_param": 42},  # type: ignore
                position={"x": 0.0, "y": 0.0},
            )

    def test_validate_ir_rejects_no_input_node(self) -> None:
        """Verify that validate_ir_graph() raises error when no input node exists."""
        # Create a graph with only a Dense node (no Input node)
        dense_node = IRNode(
            id="node-1",
            node_params=DenseParams(layer_type="dense", units=64),
            position={"x": 0.0, "y": 0.0},
        )
        graph = IRGraph(nodes=[dense_node], edges=[])

        with pytest.raises(IRValidationError, match="must contain at least one input node"):
            validate_ir_graph(graph)

    def test_validate_ir_rejects_concatenate_with_one_edge(self) -> None:
        """Verify that Concatenate nodes with fewer than 2 inputs are rejected."""
        # Create a valid input node
        input_node = IRNode(
            id="node-1",
            node_params=InputParams(layer_type="input", shape=784),
            position={"x": 0.0, "y": 0.0},
        )

        # Create a Concatenate node
        concat_node = IRNode(
            id="node-2",
            node_params=ConcatenateParams(layer_type="concatenate", axis=-1),
            position={"x": 0.0, "y": 100.0},
        )

        # Create only ONE edge to the Concatenate node (invalid)
        edge = IREdge(id="edge-1", source_id="node-1", target_id="node-2")

        graph = IRGraph(nodes=[input_node, concat_node], edges=[edge])

        with pytest.raises(IRValidationError, match="must have at least 2 incoming edges"):
            validate_ir_graph(graph)

    def test_validate_ir_accepts_valid_graph(self) -> None:
        """Verify that a valid graph passes validation."""
        # Input -> Dense -> Output
        input_node = IRNode(
            id="node-1",
            node_params=InputParams(layer_type="input", shape=784),
            position={"x": 0.0, "y": 0.0},
        )
        dense_node = IRNode(
            id="node-2",
            node_params=DenseParams(layer_type="dense", units=64),
            position={"x": 0.0, "y": 100.0},
        )
        edge = IREdge(id="edge-1", source_id="node-1", target_id="node-2")

        graph = IRGraph(nodes=[input_node, dense_node], edges=[edge])

        # Should not raise
        validate_ir_graph(graph)

    def test_validate_ir_rejects_cycle(self) -> None:
        """Verify that graphs with cycles are rejected (DAG constraint)."""
        # Create a cycle: node-1 -> node-2 -> node-1
        input_node = IRNode(
            id="node-1",
            node_params=InputParams(layer_type="input", shape=784),
            position={"x": 0.0, "y": 0.0},
        )
        dense_node = IRNode(
            id="node-2",
            node_params=DenseParams(layer_type="dense", units=64),
            position={"x": 0.0, "y": 100.0},
        )

        # Create edges that form a cycle
        edge1 = IREdge(id="edge-1", source_id="node-1", target_id="node-2")
        edge2 = IREdge(id="edge-2", source_id="node-2", target_id="node-1")

        graph = IRGraph(nodes=[input_node, dense_node], edges=[edge1, edge2])

        with pytest.raises(IRValidationError, match="contains a cycle"):
            validate_ir_graph(graph)

    def test_validate_ir_rejects_nonexistent_node_reference(self) -> None:
        """Verify that edges referencing non-existent nodes are rejected."""
        input_node = IRNode(
            id="node-1",
            node_params=InputParams(layer_type="input", shape=784),
            position={"x": 0.0, "y": 0.0},
        )

        # Edge references a non-existent target node
        edge = IREdge(id="edge-1", source_id="node-1", target_id="node-999")

        graph = IRGraph(nodes=[input_node], edges=[edge])

        with pytest.raises(IRValidationError, match="references non-existent.*node-999"):
            validate_ir_graph(graph)
