"""Tests for the static graph analyzer (shape inference, parameter counts, diagnostics).

None of these import TensorFlow. ``test_graph_analysis_parity.py`` checks the same rules against
real Keras models.
"""

import pytest

from app.ir.analysis import DiagnosticCode, Severity, analyze_canvas, analyze_graph
from app.ir.analysis.shapes import concat_shapes, registered_layer_types
from app.ir.schema import IREdge, IRGraph, IRNode
from app.layers.registry import LAYER_REGISTRY


def make_graph(nodes: list[tuple[str, dict]], edges: list[tuple[str, str]]) -> IRGraph:
    """``nodes`` are (id, node_params); ``edges`` are (source, target)."""
    return IRGraph(
        nodes=[IRNode(id=nid, node_params=params, name=nid) for nid, params in nodes],
        edges=[IREdge(id=f"{s}->{t}", source_id=s, target_id=t) for s, t in edges],
    )


def chain(*layers: tuple[str, dict]) -> IRGraph:
    ids = [nid for nid, _ in layers]
    return make_graph(list(layers), list(zip(ids, ids[1:], strict=False)))


def codes(analysis) -> list[DiagnosticCode]:
    return [d.code for d in analysis.diagnostics]


def inp(shape: int) -> tuple[str, dict]:
    return "in", {"layer_type": "input", "shape": shape}


def image_cnn() -> IRGraph:
    return chain(
        inp(784),
        ("rs", {"layer_type": "reshape", "target_shape": "28,28,1"}),
        ("c1", {"layer_type": "conv2d", "filters": 8, "kernel_size": 3, "strides": 1, "padding": "valid"}),
        ("mp", {"layer_type": "maxpool2d", "pool_size": 2, "strides": 2}),
        ("fl", {"layer_type": "flatten"}),
        ("out", {"layer_type": "dense", "units": 10, "activation": "softmax"}),
    )


class TestRuleCoverage:
    def test_every_registry_layer_has_a_rule(self):
        """Adding a layer to the registry without a shape rule must fail loudly."""
        assert registered_layer_types() == set(LAYER_REGISTRY)


class TestShapeInference:
    def test_dense_chain(self):
        a = analyze_graph(
            chain(
                inp(10),
                ("d1", {"layer_type": "dense", "units": 64}),
                ("d2", {"layer_type": "dense", "units": 3}),
            )
        )
        assert a.ok and a.complete
        assert a.node("d1").output_shape == (64,)
        assert a.node("d1").params == 10 * 64 + 64
        assert a.total_params == (10 * 64 + 64) + (64 * 3 + 3)
        assert a.output_node_ids == ["d2"]
        assert a.size_bytes == a.total_params * 4

    def test_cnn_shapes_follow_keras_arithmetic(self):
        a = analyze_graph(image_cnn())
        assert a.ok and a.complete
        assert [a.node(i).output_shape for i in ("in", "rs", "c1", "mp", "fl", "out")] == [
            (784,),
            (28, 28, 1),
            (26, 26, 8),
            (13, 13, 8),
            (1352,),
            (10,),
        ]
        assert a.node("c1").params == 3 * 3 * 1 * 8 + 8
        assert a.node("c1").macs == 26 * 26 * 3 * 3 * 1 * 8

    def test_same_padding_uses_ceil_of_size_over_stride(self):
        a = analyze_graph(
            chain(
                inp(49),
                ("rs", {"layer_type": "reshape", "target_shape": "7,7,1"}),
                ("c", {"layer_type": "conv2d", "filters": 4, "kernel_size": 3, "strides": 2, "padding": "same"}),
            )
        )
        assert a.node("c").output_shape == (4, 4, 4)

    def test_reshape_infers_minus_one(self):
        a = analyze_graph(chain(inp(12), ("rs", {"layer_type": "reshape", "target_shape": "-1,3"})))
        assert a.node("rs").output_shape == (4, 3)

    def test_recurrent_layers(self):
        a = analyze_graph(
            chain(
                inp(20),
                ("rs", {"layer_type": "reshape", "target_shape": "5,4"}),
                ("l1", {"layer_type": "lstm", "units": 8, "return_sequences": True}),
                ("g1", {"layer_type": "gru", "units": 6}),
            )
        )
        assert a.node("l1").output_shape == (5, 8)
        assert a.node("l1").params == 4 * (4 * 8 + 8 * 8 + 8)
        assert a.node("g1").output_shape == (6,)
        assert a.node("g1").params == 3 * (8 * 6 + 6 * 6 + 2 * 6)

    def test_embedding_adds_a_feature_axis(self):
        a = analyze_graph(chain(inp(30), ("e", {"layer_type": "embedding", "input_dim": 1000, "output_dim": 16})))
        assert a.node("e").output_shape == (30, 16)
        assert a.node("e").params == 16000

    def test_batchnorm_splits_trainable_and_non_trainable(self):
        a = analyze_graph(chain(inp(10), ("bn", {"layer_type": "batchnorm"})))
        bn = a.node("bn")
        assert (bn.params, bn.trainable_params, bn.non_trainable_params) == (40, 20, 20)
        assert a.non_trainable_params == 20

    def test_merging_branches_with_concatenate(self):
        graph = make_graph(
            [
                inp(8),
                ("a", {"layer_type": "dense", "units": 5}),
                ("b", {"layer_type": "dense", "units": 7}),
                ("cat", {"layer_type": "concatenate", "axis": -1}),
                ("out", {"layer_type": "dense", "units": 2}),
            ],
            [("in", "a"), ("in", "b"), ("a", "cat"), ("b", "cat"), ("cat", "out")],
        )
        a = analyze_graph(graph)
        assert a.ok
        assert a.node("cat").output_shape == (12,)
        assert a.node("cat").input_shapes == [(5,), (7,)]
        assert a.node("out").params == 12 * 2 + 2

    def test_non_merge_layer_with_two_inputs_concatenates_on_last_axis(self):
        graph = make_graph(
            [
                inp(8),
                ("a", {"layer_type": "dense", "units": 5}),
                ("b", {"layer_type": "dense", "units": 7}),
                ("out", {"layer_type": "dense", "units": 2}),
            ],
            [("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")],
        )
        a = analyze_graph(graph)
        assert a.ok
        assert a.node("out").params == 12 * 2 + 2

    def test_multiple_input_nodes_are_allowed(self):
        graph = make_graph(
            [
                ("i1", {"layer_type": "input", "shape": 4}),
                ("i2", {"layer_type": "input", "shape": 6}),
                ("cat", {"layer_type": "concatenate", "axis": -1}),
            ],
            [("i1", "cat"), ("i2", "cat")],
        )
        a = analyze_graph(graph)
        assert a.ok and a.node("cat").output_shape == (10,)

    def test_nodes_are_reported_in_topological_order(self):
        graph = make_graph(
            [("out", {"layer_type": "dense", "units": 1}), inp(4)],
            [("in", "out")],
        )
        assert [n.node_id for n in analyze_graph(graph).nodes] == ["in", "out"]


class TestConcatShapes:
    def test_axis_counts_the_batch_axis_like_keras(self):
        assert concat_shapes([(4, 3), (4, 5)], axis=2) == (4, 8)
        assert concat_shapes([(4, 3), (6, 3)], axis=1) == (10, 3)

    def test_batch_axis_is_rejected(self):
        with pytest.raises(Exception, match="batch"):
            concat_shapes([(4,), (4,)], axis=0)

    def test_axis_out_of_range(self):
        with pytest.raises(Exception, match="out of range"):
            concat_shapes([(4,), (4,)], axis=5)


class TestDiagnostics:
    def test_conv_on_flat_input_suggests_a_reshape(self):
        a = analyze_graph(
            chain(inp(784), ("c", {"layer_type": "conv2d", "filters": 8, "kernel_size": 3, "padding": "same"}))
        )
        assert not a.ok
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.RANK_MISMATCH
        assert d.node_id == "c" and d.related_node_ids == ["in"]
        assert "3-D" in d.message and "(784,)" in d.message
        assert "28,28,1" in d.suggestion

    def test_conv_suggestion_for_non_square_input_gives_the_size_to_match(self):
        a = analyze_graph(chain(inp(30), ("c", {"layer_type": "conv2d", "filters": 2})))
        assert "multiply to 30" in a.diagnostics[0].suggestion

    def test_lstm_on_flat_input_suggests_a_sequence_reshape(self):
        a = analyze_graph(chain(inp(16), ("l", {"layer_type": "lstm", "units": 4})))
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.RANK_MISMATCH
        assert "16,1" in d.suggestion

    def test_kernel_larger_than_input_with_valid_padding(self):
        a = analyze_graph(
            chain(
                inp(9),
                ("rs", {"layer_type": "reshape", "target_shape": "3,3,1"}),
                ("c", {"layer_type": "conv2d", "filters": 2, "kernel_size": 5, "padding": "valid"}),
            )
        )
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.INVALID_OUTPUT_SIZE
        assert "3x3" in d.message
        assert "same" in d.suggestion

    def test_pooling_window_larger_than_input(self):
        a = analyze_graph(
            chain(
                inp(4),
                ("rs", {"layer_type": "reshape", "target_shape": "2,2,1"}),
                ("p", {"layer_type": "maxpool2d", "pool_size": 3, "strides": 1, "padding": "valid"}),
            )
        )
        assert codes(a) == [DiagnosticCode.INVALID_OUTPUT_SIZE]

    def test_reshape_size_mismatch(self):
        a = analyze_graph(chain(inp(10), ("rs", {"layer_type": "reshape", "target_shape": "3,3"})))
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.INVALID_RESHAPE
        assert "10 values" in d.message and "9 values" in d.message
        assert "multiply to 10" in d.suggestion

    @pytest.mark.parametrize("target", ["a,b", "0,5", "-2,5", "-1,-1"])
    def test_reshape_rejects_malformed_targets(self, target):
        a = analyze_graph(chain(inp(10), ("rs", {"layer_type": "reshape", "target_shape": target})))
        assert codes(a) == [DiagnosticCode.INVALID_RESHAPE]

    def test_reshape_minus_one_must_divide_evenly(self):
        a = analyze_graph(chain(inp(10), ("rs", {"layer_type": "reshape", "target_shape": "-1,4"})))
        assert codes(a) == [DiagnosticCode.INVALID_RESHAPE]

    def test_concatenate_dimension_mismatch_names_both_branches(self):
        graph = make_graph(
            [
                inp(8),
                ("r1", {"layer_type": "reshape", "target_shape": "2,4"}),
                ("r2", {"layer_type": "reshape", "target_shape": "4,2"}),
                ("cat", {"layer_type": "concatenate", "axis": -1}),
            ],
            [("in", "r1"), ("in", "r2"), ("r1", "cat"), ("r2", "cat")],
        )
        a = analyze_graph(graph)
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.SHAPE_MISMATCH
        assert d.related_node_ids == ["r1", "r2"]
        assert "(2, 4)" in d.message and "(4, 2)" in d.message

    def test_concatenate_with_a_single_input(self):
        a = analyze_graph(chain(inp(4), ("cat", {"layer_type": "concatenate"})))
        assert codes(a) == [DiagnosticCode.MERGE_ARITY]
        assert a.diagnostics[0].message.endswith("has 1.")

    def test_one_mistake_is_reported_once_not_per_downstream_layer(self):
        a = analyze_graph(
            chain(
                inp(784),
                ("c", {"layer_type": "conv2d", "filters": 4}),
                ("mp", {"layer_type": "maxpool2d"}),
                ("fl", {"layer_type": "flatten"}),
                ("out", {"layer_type": "dense", "units": 10}),
            )
        )
        assert len(a.diagnostics) == 1 and a.diagnostics[0].node_id == "c"
        assert not a.complete
        assert a.node("out").output_shape is None
        assert a.node("in").output_shape == (784,)

    def test_dense_after_conv_is_a_warning_not_an_error(self):
        a = analyze_graph(
            chain(
                inp(16),
                ("rs", {"layer_type": "reshape", "target_shape": "4,4,1"}),
                ("c", {"layer_type": "conv2d", "filters": 2, "padding": "same"}),
                ("d", {"layer_type": "dense", "units": 3}),
            )
        )
        assert a.ok and a.complete
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.UNFLATTENED_DENSE and d.severity == Severity.WARNING
        assert "Flatten" in d.suggestion
        assert a.node("d").output_shape == (4, 4, 3)

    def test_no_input_node(self):
        a = analyze_graph(make_graph([("d", {"layer_type": "dense", "units": 2})], []))
        assert DiagnosticCode.NO_INPUT_NODE in codes(a)
        assert DiagnosticCode.DISCONNECTED_NODE in codes(a)

    def test_disconnected_node_downstream_stays_quiet(self):
        graph = make_graph(
            [
                inp(4),
                ("a", {"layer_type": "dense", "units": 2}),
                ("orphan", {"layer_type": "dense", "units": 3}),
                ("after", {"layer_type": "dense", "units": 1}),
            ],
            [("in", "a"), ("orphan", "after")],
        )
        a = analyze_graph(graph)
        assert [(d.code, d.node_id) for d in a.diagnostics] == [(DiagnosticCode.DISCONNECTED_NODE, "orphan")]

    def test_dangling_edge(self):
        graph = make_graph([inp(4), ("d", {"layer_type": "dense", "units": 2})], [("in", "d"), ("d", "ghost")])
        a = analyze_graph(graph)
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.DANGLING_EDGE and d.node_id == "d"
        assert "ghost" in d.message

    def test_cycle_reports_only_the_nodes_on_the_cycle(self):
        graph = make_graph(
            [
                inp(4),
                ("a", {"layer_type": "dense", "units": 4}),
                ("b", {"layer_type": "dense", "units": 4}),
                ("tail", {"layer_type": "dense", "units": 1}),
            ],
            [("in", "a"), ("a", "b"), ("b", "a"), ("b", "tail")],
        )
        a = analyze_graph(graph)
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.CYCLE
        assert set(d.related_node_ids) == {"a", "b"}
        assert {n.node_id for n in a.nodes} == {"in", "a", "b", "tail"}

    def test_edge_into_input_is_a_warning(self):
        graph = make_graph(
            [inp(4), ("d", {"layer_type": "dense", "units": 4})],
            [("in", "d"), ("d", "in")],
        )
        a = analyze_graph(graph)
        assert a.ok is False  # The edge pair also forms a cycle.
        graph = make_graph(
            [inp(4), ("i2", {"layer_type": "input", "shape": 4}), ("d", {"layer_type": "dense", "units": 4})],
            [("in", "d"), ("d", "i2")],
        )
        a = analyze_graph(graph)
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.INPUT_HAS_INCOMING_EDGE and d.severity == Severity.WARNING
        assert a.ok


class TestCanvasAnalysis:
    @staticmethod
    def node(node_id, ntype, **params):
        return {"id": node_id, "type": ntype, "position": {"x": 0, "y": 0}, "data": {"name": node_id, "params": params}}

    @staticmethod
    def edge(source, target):
        return {"id": f"{source}-{target}", "source": source, "target": target}

    def test_registry_style_canvas(self):
        canvas = {
            "nodes": [
                self.node("in", "inputNode", shape=10),
                self.node("d", "denseNode", units=4, activation="relu"),
            ],
            "edges": [self.edge("in", "d")],
        }
        a = analyze_canvas(canvas)
        assert a.ok and a.node("d").output_shape == (4,)

    def test_legacy_custom_node_types(self):
        canvas = {
            "nodes": [
                self.node("in", "custominput", **{"dim-1": 10}),
                self.node("d", "customdense", units=4, activation="relu"),
            ],
            "edges": [self.edge("in", "d")],
        }
        a = analyze_canvas(canvas)
        assert a.ok and a.node("d").params == 44

    def test_a_node_with_bad_params_does_not_hide_the_rest_of_the_graph(self):
        canvas = {
            "nodes": [
                self.node("in", "inputNode", shape=10),
                self.node("bad", "denseNode", units=0),
                self.node("ok", "denseNode", units=3),
            ],
            "edges": [self.edge("in", "bad"), self.edge("in", "ok")],
        }
        a = analyze_canvas(canvas)
        (d,) = a.diagnostics
        assert d.code == DiagnosticCode.INVALID_PARAMS and d.node_id == "bad"
        assert "units" in d.message
        assert a.node("ok").output_shape == (3,)
        assert a.node("bad").output_shape is None

    def test_downstream_of_an_invalid_node_is_not_reported_again(self):
        canvas = {
            "nodes": [
                self.node("in", "inputNode", shape=10),
                self.node("bad", "denseNode"),  # units missing
                self.node("next", "denseNode", units=2),
            ],
            "edges": [self.edge("in", "bad"), self.edge("bad", "next")],
        }
        a = analyze_canvas(canvas)
        assert [d.node_id for d in a.diagnostics] == ["bad"]

    def test_unknown_layer_type(self):
        a = analyze_canvas({"nodes": [self.node("x", "mysteryNode")], "edges": []})
        assert DiagnosticCode.INVALID_PARAMS in codes(a)

    def test_empty_canvas(self):
        a = analyze_canvas({"nodes": [], "edges": []})
        assert codes(a) == [DiagnosticCode.EMPTY_GRAPH]
        assert not a.ok and not a.complete

    def test_edges_with_missing_endpoints_do_not_crash(self):
        canvas = {"nodes": [self.node("in", "inputNode", shape=4)], "edges": [{"id": "e", "source": "in"}]}
        a = analyze_canvas(canvas)
        assert DiagnosticCode.DANGLING_EDGE in codes(a)


class TestAnalyzeGraphEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    URL = "/api/v1/layers/analyze-graph"

    def test_graph_ir_payload(self, client):
        graph = image_cnn().model_dump(mode="json")
        response = client.post(self.URL, json={"graph_ir": graph})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True and body["complete"] is True
        assert body["total_params"] == (3 * 3 * 8 + 8) + (13 * 13 * 8 * 10 + 10)
        cnn = next(n for n in body["nodes"] if n["node_id"] == "c1")
        assert cnn["output_shape"] == [26, 26, 8]

    def test_canvas_payload_with_a_problem_is_still_a_200(self, client):
        canvas = {
            "nodes": [
                {"id": "in", "type": "inputNode", "data": {"params": {"shape": 784}}},
                {"id": "c", "type": "conv2dNode", "data": {"params": {"filters": 8}}},
            ],
            "edges": [{"id": "e", "source": "in", "target": "c"}],
        }
        response = client.post(self.URL, json={"canvas": canvas})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        (d,) = body["diagnostics"]
        assert d["code"] == "rank_mismatch" and d["node_id"] == "c" and d["severity"] == "error"
        assert d["suggestion"]

    def test_missing_payload_is_a_400(self, client):
        assert client.post(self.URL, json={}).status_code == 400

    def test_malformed_graph_ir_is_a_422(self, client):
        response = client.post(self.URL, json={"graph_ir": {"nodes": []}})
        assert response.status_code == 422
