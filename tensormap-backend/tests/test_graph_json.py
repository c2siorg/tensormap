"""Unit tests for graph_json storage, retrieval, and auto-layout logic.

Covers:
- Fast path (graph_json IS NOT NULL) with and without positions
- Legacy fallback path (graph_json IS NULL, configs present)
- Consistent graph shape from _extract_graph helper
- Size-guard rejection for oversized payloads
- Deep-copy safety (ORM object not mutated)
- get_available_model_list excludes graph_json from response

All database interactions use MagicMock — no running DB required.
TensorFlow / flatten_json are stubbed out via sys.modules.
"""

import copy
import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy third-party modules so tests run without them installed.
_tf_stub = MagicMock()
sys.modules.setdefault("tensorflow", _tf_stub)
sys.modules.setdefault("flatten_json", MagicMock())

from app.models.ml import ModelBasic, ModelConfigs  # noqa: E402
from app.services.deep_learning import (  # noqa: E402
    _apply_auto_layout,
    _extract_graph,
    _validate_graph_size,
    get_available_model_list,
    get_model_graph_service,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_GRAPH = {
    "nodes": [
        {"id": "n0", "type": "custominput", "data": {"params": {"dim-1": "4"}}, "position": {"x": 100.0, "y": 0.0}},
        {
            "id": "n1",
            "type": "customdense",
            "data": {"params": {"units": "16", "activation": "relu"}},
            "position": {"x": 100.0, "y": 200.0},
        },
    ],
    "edges": [{"source": "n0", "target": "n1"}],
    "model_name": "test_model",
}

SAMPLE_GRAPH_NO_POSITIONS = {
    "nodes": [
        {"id": "n0", "type": "custominput", "data": {"params": {"dim-1": "4"}}},
        {"id": "n1", "type": "customdense", "data": {"params": {"units": "16", "activation": "relu"}}},
    ],
    "edges": [{"source": "n0", "target": "n1"}],
    "model_name": "test_model",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    return MagicMock()


def _make_model(*, graph_json=None, model_name="test_model", model_id=1):
    """Create a MagicMock that behaves like a ModelBasic with the given graph_json."""
    m = MagicMock(spec=ModelBasic)
    m.id = model_id
    m.model_name = model_name
    m.graph_json = graph_json
    return m


# ---------------------------------------------------------------------------
# _extract_graph
# ---------------------------------------------------------------------------


class TestExtractGraph:
    def test_returns_none_for_none(self):
        assert _extract_graph(None) is None

    def test_unwraps_model_key(self):
        """When payload is the full request body with a 'model' sub-key, extract it."""
        payload = {"model": {"nodes": [1, 2], "edges": []}, "code": {}}
        result = _extract_graph(payload)
        assert result == {"nodes": [1, 2], "edges": []}

    def test_returns_payload_when_nodes_present(self):
        """When payload already has 'nodes' at top level, return as-is."""
        payload = {"nodes": [1, 2], "edges": [], "model_name": "m"}
        result = _extract_graph(payload)
        assert result is payload

    def test_model_validate_and_save_produce_same_shape(self):
        """Both write-path payloads should produce the same extracted graph shape."""
        graph = {"nodes": [{"id": "n0"}], "edges": [], "model_name": "m"}

        # model_validate_service receives full request body
        validate_payload = {"model": graph, "code": {"dl_model": {}}}
        # model_save_service receives request.model.model_dump() — the graph itself
        save_payload = graph

        assert _extract_graph(validate_payload) == _extract_graph(save_payload)


# ---------------------------------------------------------------------------
# _validate_graph_size
# ---------------------------------------------------------------------------


class TestValidateGraphSize:
    def test_none_graph_returns_none(self):
        assert _validate_graph_size(None) is None

    def test_small_graph_returns_none(self):
        assert _validate_graph_size(SAMPLE_GRAPH) is None

    def test_oversized_graph_returns_error(self):
        huge = {"nodes": [{"id": str(i), "data": "x" * 10000} for i in range(100)]}
        result = _validate_graph_size(huge)
        assert result is not None
        assert "too large" in result.lower()


# ---------------------------------------------------------------------------
# _apply_auto_layout
# ---------------------------------------------------------------------------


class TestApplyAutoLayout:
    def test_adds_positions_to_nodes_without_them(self):
        graph = copy.deepcopy(SAMPLE_GRAPH_NO_POSITIONS)
        _apply_auto_layout(graph)
        for node in graph["nodes"]:
            assert "position" in node
            assert isinstance(node["position"]["x"], float)
            assert isinstance(node["position"]["y"], float)

    def test_preserves_existing_positions(self):
        graph = copy.deepcopy(SAMPLE_GRAPH)
        original_positions = [copy.deepcopy(n["position"]) for n in graph["nodes"]]
        _apply_auto_layout(graph)
        for node, orig_pos in zip(graph["nodes"], original_positions, strict=True):
            assert node["position"] == orig_pos

    def test_handles_empty_nodes(self):
        graph = {"nodes": [], "edges": []}
        _apply_auto_layout(graph)  # should not raise

    def test_handles_missing_nodes_key(self):
        graph = {"edges": []}
        _apply_auto_layout(graph)  # should not raise


# ---------------------------------------------------------------------------
# get_model_graph_service — fast path
# ---------------------------------------------------------------------------


class TestGetModelGraphFastPath:
    def test_returns_graph_from_graph_json(self, mock_db):
        """When graph_json is populated, it should be returned directly."""
        model = _make_model(graph_json=copy.deepcopy(SAMPLE_GRAPH))
        mock_db.exec.return_value.first.return_value = model

        body, status = get_model_graph_service(mock_db, "test_model")

        assert status == 200
        assert body["success"] is True
        assert body["data"]["graph"]["nodes"] == SAMPLE_GRAPH["nodes"]
        assert body["data"]["model_name"] == "test_model"

    def test_adds_positions_when_missing(self, mock_db):
        """Fast path should auto-layout nodes that lack positions."""
        model = _make_model(graph_json=copy.deepcopy(SAMPLE_GRAPH_NO_POSITIONS))
        mock_db.exec.return_value.first.return_value = model

        body, status = get_model_graph_service(mock_db, "test_model")

        assert status == 200
        for node in body["data"]["graph"]["nodes"]:
            assert "position" in node

    def test_does_not_mutate_orm_object(self, mock_db):
        """The service must not modify model.graph_json in-place."""
        original_graph = copy.deepcopy(SAMPLE_GRAPH_NO_POSITIONS)
        model = _make_model(graph_json=copy.deepcopy(original_graph))
        mock_db.exec.return_value.first.return_value = model

        get_model_graph_service(mock_db, "test_model")

        # The ORM object's graph_json should be unchanged (no positions added)
        for node in model.graph_json["nodes"]:
            assert "position" not in node

    def test_model_not_found_returns_404(self, mock_db):
        mock_db.exec.return_value.first.return_value = None

        body, status = get_model_graph_service(mock_db, "nonexistent")

        assert status == 404
        assert body["success"] is False


# ---------------------------------------------------------------------------
# get_model_graph_service — legacy fallback path
# ---------------------------------------------------------------------------


class TestGetModelGraphLegacyPath:
    def test_falls_back_to_model_configs(self, mock_db):
        """When graph_json is None, the service reconstructs from ModelConfigs."""
        model = _make_model(graph_json=None)

        # First .exec().first() returns the model; second .exec().all() returns configs
        first_call = MagicMock()
        first_call.first.return_value = model

        cfg1 = MagicMock(spec=ModelConfigs)
        cfg1.parameter = "nodes.0.id"
        cfg1.value = "n0"
        cfg2 = MagicMock(spec=ModelConfigs)
        cfg2.parameter = "nodes.0.type"
        cfg2.value = "custominput"
        cfg3 = MagicMock(spec=ModelConfigs)
        cfg3.parameter = "nodes.0.data.params.dim-1"
        cfg3.value = "4"

        second_call = MagicMock()
        second_call.all.return_value = [cfg1, cfg2, cfg3]

        mock_db.exec.side_effect = [first_call, second_call]

        body, status = get_model_graph_service(mock_db, "test_model")

        assert status == 200
        assert body["success"] is True
        graph = body["data"]["graph"]
        assert "nodes" in graph
        # Auto-layout should have been applied
        for node in graph["nodes"]:
            assert "position" in node

    def test_legacy_no_configs_returns_404(self, mock_db):
        """When graph_json is None AND no ModelConfigs exist, return 404."""
        model = _make_model(graph_json=None)

        first_call = MagicMock()
        first_call.first.return_value = model

        second_call = MagicMock()
        second_call.all.return_value = []

        mock_db.exec.side_effect = [first_call, second_call]

        body, status = get_model_graph_service(mock_db, "test_model")

        assert status == 404
        assert body["success"] is False


# ---------------------------------------------------------------------------
# get_available_model_list — should NOT include graph_json
# ---------------------------------------------------------------------------


class TestGetAvailableModelList:
    def test_excludes_graph_json_from_response(self, mock_db):
        """The list endpoint should return only id and model_name."""
        m1 = MagicMock(spec=ModelBasic)
        m1.id = 1
        m1.model_name = "model_a"
        m1.graph_json = {"nodes": [], "edges": []}

        m2 = MagicMock(spec=ModelBasic)
        m2.id = 2
        m2.model_name = "model_b"
        m2.graph_json = None

        # count query
        mock_db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=2)),  # total count
            MagicMock(all=MagicMock(return_value=[m1, m2])),  # model list
        ]

        body, status = get_available_model_list(mock_db)

        assert status == 200
        for item in body["data"]:
            assert "graph_json" not in item
            assert "id" in item
            assert "model_name" in item
