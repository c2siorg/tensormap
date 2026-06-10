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
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ir.schema import (
    ConcatenateParams,
    DenseParams,
    InputParams,
    IREdge,
    IRGraph,
    IRNode,
    validate_ir_graph,
)
from app.ir.translator import ir_to_keras_build_args, reactflow_to_ir
from app.layers.registry import (
    LAYER_CATEGORIES,
    LAYER_REGISTRY,
    ParamType,
    get_layer_spec,
    serialize_registry,
)
from app.main import app


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
        """Verify that validate_ir_graph() returns error when no input node exists."""
        # Create a graph with only a Dense node (no Input node)
        dense_node = IRNode(
            id="node-1",
            node_params=DenseParams(layer_type="dense", units=64),
            position={"x": 0.0, "y": 0.0},
        )
        graph = IRGraph(nodes=[dense_node], edges=[])

        errors = validate_ir_graph(graph)
        assert len(errors) > 0
        assert any("input node" in err.message.lower() for err in errors)

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

        errors = validate_ir_graph(graph)
        assert len(errors) > 0
        assert any("at least 2 incoming edges" in err.message for err in errors)

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

        # Should return no errors
        errors = validate_ir_graph(graph)
        assert len(errors) == 0

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

        errors = validate_ir_graph(graph)
        assert len(errors) > 0
        assert any("cycle" in err.message.lower() for err in errors)

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

        errors = validate_ir_graph(graph)
        assert len(errors) > 0
        assert any("node-999" in err.message for err in errors)


client = TestClient(app)


def test_get_layers_endpoint():
    resp = client.get("/api/v1/layers/")
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert "layers" in data


def test_get_layers_contains_all_15():
    resp = client.get("/api/v1/layers/")
    data = resp.json()
    count = sum(len(layers) for layers in data["layers"].values())
    assert count >= 15


def test_get_layers_categories_ordered():
    resp = client.get("/api/v1/layers/")
    data = resp.json()
    assert data["categories"] == LAYER_CATEGORIES


def test_ir_translator_dense_roundtrip():
    rf_json = {
        "nodes": [
            {
                "id": "n1",
                "type": "denseNode",
                "position": {"x": 0, "y": 0},
                "data": {"name": "Dense", "params": {"layer_type": "dense", "units": 10, "activation": "relu"}},
            }
        ],
        "edges": [],
    }
    graph = reactflow_to_ir(rf_json)
    out_rf = graph.to_reactflow_json()

    assert len(out_rf["nodes"]) == 1
    # Note: to_reactflow_json strips layer_type internally when creating params mapping
    params = out_rf["nodes"][0]["data"]["params"]
    assert "units" in params
    assert params["units"] == 10


def test_ir_translator_lstm():
    rf_json = {
        "nodes": [
            {
                "id": "n1",
                "type": "lstmNode",
                "position": {"x": 0, "y": 0},
                "data": {"name": "LSTM", "params": {"layer_type": "lstm", "units": 20, "return_sequences": True}},
            }
        ],
        "edges": [],
    }
    graph = reactflow_to_ir(rf_json)
    params = graph.nodes[0].node_params
    assert params.return_sequences is True


def test_ir_build_args_dropout():
    rf_json = {
        "nodes": [{"id": "n1", "type": "dropoutNode", "data": {"params": {"layer_type": "dropout", "rate": 0.5}}}]
    }
    graph = reactflow_to_ir(rf_json)
    kwargs = ir_to_keras_build_args(graph.nodes[0])
    assert kwargs.get("rate") == 0.5


def test_ir_build_args_concatenate_has_axis():
    rf_json = {
        "nodes": [
            {"id": "n1", "type": "concatenateNode", "data": {"params": {"layer_type": "concatenate", "axis": -1}}}
        ]
    }
    graph = reactflow_to_ir(rf_json)
    kwargs = ir_to_keras_build_args(graph.nodes[0])
    assert kwargs.get("axis") == -1


def test_get_layer_spec_raises_keyerror():
    with pytest.raises(KeyError):
        _ = LAYER_REGISTRY["unknown"]


def test_ir_graph_validates_on_construction():
    with pytest.raises(ValidationError):
        DenseParams(layer_type="dense", units=0)


def test_concatenate_input_node_ids_ordering():
    rf_json = {
        "nodes": [
            {"id": "in1", "type": "inputNode", "data": {"params": {"layer_type": "input", "shape": 10}}},
            {"id": "in2", "type": "inputNode", "data": {"params": {"layer_type": "input", "shape": 10}}},
            {"id": "concat", "type": "concatenateNode", "data": {"params": {"layer_type": "concatenate"}}},
        ],
        "edges": [{"id": "e1", "source": "in1", "target": "concat"}, {"id": "e2", "source": "in2", "target": "concat"}],
    }
    graph = reactflow_to_ir(rf_json)
    c_node = next(n for n in graph.nodes if getattr(n.node_params, "layer_type", "") == "concatenate")
    assert c_node.input_node_ids == ["in1", "in2"]
