import pytest
from fastapi.testclient import TestClient

from app.ir.schema import ConcatenateParams, DenseParams, InputParams, IREdge, IRGraph, IRNode, validate_ir_graph
from app.ir.translator import TranslationError, translate_params_to_ir
from app.main import app

client = TestClient(app)


def test_valid_dense_graph_passes():
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="n1", node_params=InputParams(layer_type="input", shape=10), name="in"),
            IRNode(id="n2", node_params=DenseParams(layer_type="dense", units=10, activation="relu"), name="d"),
        ],
        edges=[IREdge(id="e1", source_id="n1", target_id="n2")],
    )
    errors = validate_ir_graph(graph)
    assert not errors


def test_valid_cnn_graph_passes():
    # just create a simple graph with input, no cycles
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="n1", node_params=InputParams(layer_type="input", shape=10), name="in"),
            IRNode(id="n2", node_params=DenseParams(layer_type="dense", units=10, activation="relu"), name="d"),
        ],
        edges=[],
    )
    errors = validate_ir_graph(graph)
    assert not errors


def test_valid_rnn_graph_passes():
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="n1", node_params=InputParams(layer_type="input", shape=10), name="in"),
        ],
        edges=[],
    )
    errors = validate_ir_graph(graph)
    assert not errors


def test_cycle_detection():
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="n1", node_params=InputParams(layer_type="input", shape=10), name="in"),
            IRNode(id="n2", node_params=DenseParams(layer_type="dense", units=10, activation="relu"), name="d"),
        ],
        edges=[IREdge(id="e1", source_id="n1", target_id="n2"), IREdge(id="e2", source_id="n2", target_id="n1")],
    )
    errors = validate_ir_graph(graph)
    assert any("cycle" in err.message for err in errors)


def test_missing_input_node():
    graph = IRGraph(
        version="1.0",
        nodes=[IRNode(id="n1", node_params=DenseParams(layer_type="dense", units=10, activation="relu"), name="d")],
        edges=[],
    )
    errors = validate_ir_graph(graph)
    assert any("input node" in err.message.lower() for err in errors)


def test_concatenate_one_edge_fails():
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="n1", node_params=InputParams(layer_type="input", shape=10), name="in"),
            IRNode(id="n2", node_params=ConcatenateParams(layer_type="concatenate"), name="c", input_node_ids=["n1"]),
        ],
        edges=[IREdge(id="e1", source_id="n1", target_id="n2")],
    )
    errors = validate_ir_graph(graph)
    assert any("at least 2" in err.message.lower() for err in errors)


def test_concatenate_two_edges_passes():
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="n1", node_params=InputParams(layer_type="input", shape=10), name="in"),
            IRNode(id="n2", node_params=DenseParams(layer_type="dense", units=10), name="d"),
            IRNode(
                id="n3", node_params=ConcatenateParams(layer_type="concatenate"), name="c", input_node_ids=["n1", "n2"]
            ),
        ],
        edges=[IREdge(id="e1", source_id="n1", target_id="n3"), IREdge(id="e2", source_id="n2", target_id="n3")],
    )
    errors = validate_ir_graph(graph)
    assert not errors


def test_validate_graph_endpoint_200_valid():
    graph_ir = {
        "version": "1.0",
        "nodes": [{"id": "n1", "node_params": {"layer_type": "input", "shape": 10}, "input_node_ids": []}],
        "edges": [],
    }
    resp = client.post("/api/v1/layers/validate-graph", json={"graph_ir": graph_ir})
    assert resp.status_code == 200
    assert resp.json() == {"valid": True}


def test_validate_graph_endpoint_400_invalid():
    graph_ir = {
        "version": "1.0",
        "nodes": [{"id": "n1", "node_params": {"layer_type": "input", "shape": 10}, "input_node_ids": []}],
        "edges": [{"id": "e1", "source_id": "n1", "target_id": "n1"}],
    }
    resp = client.post("/api/v1/layers/validate-graph", json={"graph_ir": graph_ir})
    assert resp.status_code == 400  # Will return {'valid': False, errors: [...]}
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0


def test_translate_params_invalid_units():
    with pytest.raises(TranslationError):
        translate_params_to_ir("dense", {"units": -1})


def test_translate_params_unknown_activation():
    with pytest.raises(TranslationError):
        translate_params_to_ir("dense", {"units": 10, "activation": "unknown_activation_fn"})


def test_get_single_layer_200():
    resp = client.get("/api/v1/layers/dense")
    assert resp.status_code == 200
    assert resp.json()["type_key"] == "dense"


def test_get_single_layer_404():
    resp = client.get("/api/v1/layers/unknown_layer")
    assert resp.status_code == 404


def test_valid_multi_input_graph_passes():
    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="in1", node_params=InputParams(layer_type="input", shape=10), name="in1"),
            IRNode(id="in2", node_params=InputParams(layer_type="input", shape=20), name="in2"),
            IRNode(
                id="cat",
                node_params=ConcatenateParams(layer_type="concatenate", axis=-1),
                name="cat",
                input_node_ids=["in1", "in2"],
            ),
            IRNode(id="out", node_params=DenseParams(layer_type="dense", units=5), name="out"),
        ],
        edges=[
            IREdge(id="e1", source_id="in1", target_id="cat"),
            IREdge(id="e2", source_id="in2", target_id="cat"),
            IREdge(id="e3", source_id="cat", target_id="out"),
        ],
    )
    errors = validate_ir_graph(graph)
    assert not errors


def test_validate_graph_endpoint_200_multi_input():
    graph_ir = {
        "version": "1.0",
        "nodes": [
            {"id": "in1", "node_params": {"layer_type": "input", "shape": 10}, "input_node_ids": []},
            {"id": "in2", "node_params": {"layer_type": "input", "shape": 20}, "input_node_ids": []},
            {
                "id": "cat",
                "node_params": {"layer_type": "concatenate", "axis": -1},
                "input_node_ids": ["in1", "in2"],
            },
        ],
        "edges": [
            {"id": "e1", "source_id": "in1", "target_id": "cat"},
            {"id": "e2", "source_id": "in2", "target_id": "cat"},
        ],
    }
    resp = client.post("/api/v1/layers/validate-graph", json={"graph_ir": graph_ir})
    assert resp.status_code == 200
    assert resp.json() == {"valid": True}


def test_blocking_ir_errors_accepts_multi_input():
    from app.services.deep_learning import _blocking_ir_errors

    graph = IRGraph(
        version="1.0",
        nodes=[
            IRNode(id="in1", node_params=InputParams(layer_type="input", shape=10), name="in1"),
            IRNode(id="in2", node_params=InputParams(layer_type="input", shape=20), name="in2"),
            IRNode(
                id="cat",
                node_params=ConcatenateParams(layer_type="concatenate", axis=-1),
                name="cat",
                input_node_ids=["in1", "in2"],
            ),
        ],
        edges=[
            IREdge(id="e1", source_id="in1", target_id="cat"),
            IREdge(id="e2", source_id="in2", target_id="cat"),
        ],
    )
    assert _blocking_ir_errors(graph) == []
