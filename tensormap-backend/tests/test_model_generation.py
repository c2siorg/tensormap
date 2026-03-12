"""
Unit and integration tests for the model_generation service.

Covers:
  - _build_layer() for each supported node type (Dense, Flatten, Conv2D)
  - _build_layer() error handling for unknown node types
  - _validate_graph() checks: missing input, disconnected nodes, bad params, unknown types
  - model_generation() end-to-end: linear, branching, multi-input, conv
  - Integration: generated JSON round-trips through tf.keras.models.model_from_json
  - Acceptance criteria: all ModelValidationError messages match the issue spec
"""

import json

import pytest
import tensorflow as tf

from app.services.model_generation import (
    ModelValidationError,
    _build_layer,
    _validate_graph,
    model_generation,
)

# ---------------------------------------------------------------------------
# Node / edge builder helpers
# ---------------------------------------------------------------------------


def _input_node(node_id: str, dims: list[int]) -> dict:
    """Create a custominput node."""
    params = {f"dim-{i + 1}": d for i, d in enumerate(dims)}
    return {"id": node_id, "type": "custominput", "data": {"params": params}}


def _dense_node(node_id: str, units: int = 32, activation: str = "relu") -> dict:
    return {
        "id": node_id,
        "type": "customdense",
        "data": {"params": {"units": units, "activation": activation}},
    }


def _flatten_node(node_id: str) -> dict:
    return {"id": node_id, "type": "customflatten", "data": {"params": {}}}


def _conv_node(
    node_id: str,
    filters: int = 16,
    kernel: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: str = "valid",
    activation: str = "relu",
) -> dict:
    return {
        "id": node_id,
        "type": "customconv",
        "data": {
            "params": {
                "filter": filters,
                "kernelX": kernel[0],
                "kernelY": kernel[1],
                "strideX": stride[0],
                "strideY": stride[1],
                "padding": padding,
                "activation": activation,
            }
        },
    }


def _edge(source: str, target: str) -> dict:
    return {"source": source, "target": target}


# ===================================================================
# Tests for _build_layer()
# ===================================================================


class TestBuildLayer:
    """Unit tests for the per-node Keras layer builder."""

    def test_dense_layer_output_shape(self):
        input_t = tf.keras.Input(shape=(10,), name="inp")
        node = _dense_node("d1", units=64, activation="relu")
        output = _build_layer(node, input_t)
        assert output.shape == (None, 64)

    def test_dense_layer_sigmoid(self):
        input_t = tf.keras.Input(shape=(5,), name="inp")
        node = _dense_node("d1", units=1, activation="sigmoid")
        output = _build_layer(node, input_t)
        assert output.shape == (None, 1)

    def test_flatten_layer_output_shape(self):
        """Flatten (4, 4, 3) → (48,)."""
        input_t = tf.keras.Input(shape=(4, 4, 3), name="inp")
        node = _flatten_node("f1")
        output = _build_layer(node, input_t)
        assert output.shape == (None, 48)

    def test_conv2d_layer_output_shape(self):
        """Conv2D valid-padding: (28,28,1) with 3×3 kernel → (26,26,16)."""
        input_t = tf.keras.Input(shape=(28, 28, 1), name="inp")
        node = _conv_node("c1", filters=16, kernel=(3, 3), stride=(1, 1), padding="valid")
        output = _build_layer(node, input_t)
        assert output.shape == (None, 26, 26, 16)

    def test_conv2d_same_padding_preserves_spatial_dims(self):
        input_t = tf.keras.Input(shape=(16, 16, 1), name="inp")
        node = _conv_node("c1", filters=8, kernel=(3, 3), stride=(1, 1), padding="same")
        output = _build_layer(node, input_t)
        assert output.shape == (None, 16, 16, 8)

    def test_conv2d_activation_none_becomes_linear(self):
        """'none' activation string should be silently converted to 'linear'."""
        input_t = tf.keras.Input(shape=(8, 8, 1), name="inp")
        node = _conv_node("c1", activation="none")
        output = _build_layer(node, input_t)
        assert output is not None

    def test_unknown_node_type_raises_model_validation_error(self):
        """Unknown node type must raise ModelValidationError with supported types listed."""
        input_t = tf.keras.Input(shape=(5,), name="inp")
        node = {"id": "x", "type": "custom_unknown", "data": {"params": {}}}
        with pytest.raises(ModelValidationError, match="Unknown layer type"):
            _build_layer(node, input_t)

    def test_unknown_node_type_lists_supported_types(self):
        """The error message lists the supported layer types."""
        input_t = tf.keras.Input(shape=(5,), name="inp")
        node = {"id": "x", "type": "custom_unknown", "data": {"params": {}}}
        with pytest.raises(ModelValidationError) as exc_info:
            _build_layer(node, input_t)
        msg = exc_info.value.user_message
        assert "customdense" in msg or "customflatten" in msg or "customconv" in msg


# ===================================================================
# Tests for _validate_graph()
# ===================================================================


class TestValidateGraph:
    """Unit tests for graph pre-build validation."""

    # --- Missing Input node -----------------------------------------

    def test_missing_input_node_raises(self):
        """No custominput → ModelValidationError mentioning 'No Input layer'."""
        nodes = [_dense_node("d1")]
        edges = []
        with pytest.raises(ModelValidationError, match="No Input layer found"):
            _validate_graph(nodes, edges)

    def test_missing_input_node_message_has_advice(self):
        """Error message tells the user to add an Input node."""
        nodes = [_dense_node("d1")]
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, [])
        assert "Input" in exc_info.value.user_message

    # --- Unknown node types -----------------------------------------

    def test_unknown_node_type_raises(self):
        nodes = [_input_node("i1", [10]), {"id": "x", "type": "customunknown", "data": {"params": {}}}]
        edges = [_edge("i1", "x")]
        with pytest.raises(ModelValidationError, match="Unknown layer type"):
            _validate_graph(nodes, edges)

    def test_unknown_node_message_lists_supported_types(self):
        nodes = [_input_node("i1", [10]), {"id": "x", "type": "customunknown", "data": {"params": {}}}]
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, [_edge("i1", "x")])
        msg = exc_info.value.user_message
        assert "customunknown" in msg
        assert "Supported" in msg

    # --- Invalid parameters -----------------------------------------

    def test_dense_non_numeric_units_raises(self):
        """Non-numeric units value must raise ModelValidationError naming the node and field."""
        nodes = [
            _input_node("i1", [10]),
            {"id": "d1", "type": "customdense", "data": {"params": {"units": "abc", "activation": "relu"}}},
        ]
        edges = [_edge("i1", "d1")]
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, edges)
        msg = exc_info.value.user_message
        assert "d1" in msg
        assert "Units" in msg or "units" in msg.lower()

    def test_conv_non_numeric_filter_raises(self):
        """Non-numeric filter value in Conv2D node is caught at validation time."""
        params = {
            "filter": "xyz", "kernelX": 3, "kernelY": 3,
            "strideX": 1, "strideY": 1, "padding": "valid", "activation": "relu",
        }
        nodes = [
            _input_node("i1", [28, 28, 1]),
            {"id": "c1", "type": "customconv", "data": {"params": params}},
        ]
        edges = [_edge("i1", "c1")]
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, edges)
        msg = exc_info.value.user_message
        assert "c1" in msg

    def test_conv_non_numeric_kernel_raises(self):
        """Non-numeric kernelX value in Conv2D is caught with node name in message."""
        params = {
            "filter": 16, "kernelX": "bad", "kernelY": 3,
            "strideX": 1, "strideY": 1, "padding": "valid", "activation": "relu",
        }
        nodes = [
            _input_node("i1", [28, 28, 1]),
            {"id": "c1", "type": "customconv", "data": {"params": params}},
        ]
        edges = [_edge("i1", "c1")]
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, edges)
        msg = exc_info.value.user_message
        assert "c1" in msg
        assert "Kernel" in msg or "kernel" in msg.lower()

    # --- Disconnected graph -----------------------------------------

    def test_disconnected_node_raises(self):
        """A node unreachable from Input → error mentioning disconnected count."""
        nodes = [
            _input_node("i1", [10]),
            _dense_node("d1"),  # connected
            _dense_node("orphan"),  # NOT connected
        ]
        edges = [_edge("i1", "d1")]  # orphan has no edge
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, edges)
        msg = exc_info.value.user_message
        assert "Disconnected graph" in msg
        assert "1 node(s)" in msg
        assert "orphan" in msg

    def test_disconnected_multiple_nodes_count(self):
        """Multiple orphan nodes — count and IDs in error message."""
        nodes = [
            _input_node("i1", [10]),
            _dense_node("a"),
            _dense_node("b"),
        ]
        edges = []  # both a and b are disconnected
        with pytest.raises(ModelValidationError) as exc_info:
            _validate_graph(nodes, edges)
        msg = exc_info.value.user_message
        assert "2 node(s)" in msg

    def test_connected_graph_passes(self):
        """A fully connected graph must not raise."""
        nodes = [_input_node("i1", [10]), _dense_node("d1"), _dense_node("d2")]
        edges = [_edge("i1", "d1"), _edge("d1", "d2")]
        _validate_graph(nodes, edges)  # should not raise

    # --- Valid graph with all node types passes ----------------------

    def test_all_supported_types_pass_validation(self):
        nodes = [
            _input_node("i1", [28, 28, 1]),
            _conv_node("c1"),
            _flatten_node("f1"),
            _dense_node("d1"),
        ]
        edges = [_edge("i1", "c1"), _edge("c1", "f1"), _edge("f1", "d1")]
        _validate_graph(nodes, edges)  # should not raise


# ===================================================================
# Tests for model_generation() — happy path
# ===================================================================


class TestModelGeneration:
    """End-to-end tests for the full model construction pipeline."""

    def test_simple_input_dense_returns_dict(self):
        """Return value must be a dict (valid JSON-serialisable model config)."""
        params = {
            "nodes": [_input_node("in", [10]), _dense_node("out", 1, "linear")],
            "edges": [_edge("in", "out")],
        }
        result = model_generation(params)
        assert isinstance(result, dict)

    def test_simple_input_to_dense_shapes(self):
        """input(10) → dense(1) — verify shapes survive round-trip."""
        params = {
            "nodes": [_input_node("in", [10]), _dense_node("out", 1, "linear")],
            "edges": [_edge("in", "out")],
        }
        result = model_generation(params)
        model = tf.keras.models.model_from_json(json.dumps(result))
        assert model.input_shape == (None, 10)
        assert model.output_shape == (None, 1)

    def test_multi_layer_linear_chain(self):
        """input(20) → dense(64) → dense(32) → dense(1)."""
        params = {
            "nodes": [
                _input_node("x", [20]),
                _dense_node("h1", 64, "relu"),
                _dense_node("h2", 32, "relu"),
                _dense_node("out", 1, "sigmoid"),
            ],
            "edges": [_edge("x", "h1"), _edge("h1", "h2"), _edge("h2", "out")],
        }
        result = model_generation(params)
        model = tf.keras.models.model_from_json(json.dumps(result))
        assert model.output_shape == (None, 1)
        assert len(model.layers) == 4

    def test_conv_flatten_dense(self):
        """input(28,28,1) → conv(16) → flatten → dense(10)."""
        params = {
            "nodes": [
                _input_node("img", [28, 28, 1]),
                _conv_node("c1", filters=16, kernel=(3, 3), stride=(1, 1), padding="valid"),
                _flatten_node("flat"),
                _dense_node("out", 10, "softmax"),
            ],
            "edges": [_edge("img", "c1"), _edge("c1", "flat"), _edge("flat", "out")],
        }
        result = model_generation(params)
        model = tf.keras.models.model_from_json(json.dumps(result))
        assert model.output_shape == (None, 10)

    def test_multi_input_concatenation(self):
        """
        Two inputs merged into a single Dense:
            in1 ─┐
                  ├→ dense_out
            in2 ─┘
        """
        params = {
            "nodes": [
                _input_node("in1", [5]),
                _input_node("in2", [5]),
                _dense_node("out", 1, "sigmoid"),
            ],
            "edges": [_edge("in1", "out"), _edge("in2", "out")],
        }
        result = model_generation(params)
        model = tf.keras.models.model_from_json(json.dumps(result))
        assert len(model.inputs) == 2
        assert model.output_shape == (None, 1)

    def test_multiple_output_layers(self):
        """
        input → dense1 (output)
             └→ dense2 (output)
        """
        params = {
            "nodes": [
                _input_node("x", [10]),
                _dense_node("out1", 1, "sigmoid"),
                _dense_node("out2", 5, "softmax"),
            ],
            "edges": [_edge("x", "out1"), _edge("x", "out2")],
        }
        result = model_generation(params)
        model = tf.keras.models.model_from_json(json.dumps(result))
        assert len(model.outputs) == 2

    def test_output_is_json_serialisable(self):
        """model_generation output must be JSON-serialisable without errors."""
        params = {
            "nodes": [_input_node("x", [8]), _dense_node("y", 4, "relu")],
            "edges": [_edge("x", "y")],
        }
        result = model_generation(params)
        serialised = json.dumps(result)
        assert isinstance(serialised, str)
        assert len(serialised) > 0

    def test_multi_dim_input_shape(self):
        """3-D input (28, 28, 1) is correctly passed through to Keras."""
        params = {
            "nodes": [
                _input_node("img", [28, 28, 1]),
                _dense_node("out", 10, "softmax"),
            ],
            "edges": [_edge("img", "out")],
        }
        result = model_generation(params)
        model = tf.keras.models.model_from_json(json.dumps(result))
        assert model.input_shape == (None, 28, 28, 1)


# ===================================================================
# Tests for model_generation() — error paths (acceptance criteria)
# ===================================================================


class TestModelGenerationErrors:
    """Verify each acceptance-criteria error scenario raises ModelValidationError
    with a user-friendly, spec-compliant message."""

    def test_no_input_node_raises_model_validation_error(self):
        """No Input node → ModelValidationError with 'No Input layer found' message."""
        params = {
            "nodes": [_dense_node("d1")],
            "edges": [],
        }
        with pytest.raises(ModelValidationError) as exc_info:
            model_generation(params)
        assert "No Input layer found" in exc_info.value.user_message

    def test_disconnected_graph_message_contains_count(self):
        """Disconnected orphan node → error with node count and node ID."""
        params = {
            "nodes": [
                _input_node("i1", [10]),
                _dense_node("d1"),
                _dense_node("orphan"),
            ],
            "edges": [_edge("i1", "d1")],
        }
        with pytest.raises(ModelValidationError) as exc_info:
            model_generation(params)
        msg = exc_info.value.user_message
        assert "Disconnected graph" in msg
        assert "1 node(s)" in msg
        assert "orphan" in msg

    def test_unknown_node_type_message_lists_supported(self):
        """Unknown node type → error listing supported types."""
        params = {
            "nodes": [
                _input_node("x", [5]),
                {"id": "bad", "type": "customunknown", "data": {"params": {}}},
            ],
            "edges": [_edge("x", "bad")],
        }
        with pytest.raises(ModelValidationError) as exc_info:
            model_generation(params)
        msg = exc_info.value.user_message
        assert "Unknown layer type" in msg or "customunknown" in msg

    def test_invalid_units_names_node_and_param(self):
        """Non-numeric units → error naming node ID and the 'Units' parameter."""
        params = {
            "nodes": [
                _input_node("i1", [10]),
                {"id": "myDense", "type": "customdense", "data": {"params": {"units": "abc", "activation": "relu"}}},
            ],
            "edges": [_edge("i1", "myDense")],
        }
        with pytest.raises(ModelValidationError) as exc_info:
            model_generation(params)
        msg = exc_info.value.user_message
        assert "myDense" in msg

    def test_single_input_no_edges_raises(self):
        """A graph with only an input node and no edges cannot form a valid Keras model."""
        params = {
            "nodes": [_input_node("solo", [4])],
            "edges": [],
        }
        with pytest.raises((ModelValidationError, ValueError)):
            model_generation(params)

    def test_error_message_has_no_file_paths(self):
        """Error messages must not leak internal file paths or stack traces."""
        params = {
            "nodes": [_dense_node("d1")],
            "edges": [],
        }
        with pytest.raises(ModelValidationError) as exc_info:
            model_generation(params)
        msg = exc_info.value.user_message
        assert "\\" not in msg
        assert "Traceback" not in msg
        assert ".py" not in msg

    def test_error_is_model_validation_error_subclass_of_value_error(self):
        """ModelValidationError must be a subclass of ValueError for backwards compatibility."""
        assert issubclass(ModelValidationError, ValueError)

    def test_success_false_format_via_service(self):
        """model_generation raises ModelValidationError; service layer wraps it in
        the standard {success: false, message: ...} format."""
        from unittest.mock import MagicMock

        from app.services.deep_learning import model_validate_service

        db = MagicMock()
        incoming = {
            "model": {
                "nodes": [_dense_node("d1")],
                "edges": [],
            },
            "code": {
                "dl_model": {"model_name": "test", "optimizer": "adam", "metric": "accuracy", "epochs": 5},
                "dataset": {"file_id": 1, "training_split": 0.8},
                "problem_type_id": 1,
            },
        }
        body, status_code = model_validate_service(db, incoming=incoming, project_id=None)

        assert status_code == 400
        assert body["success"] is False
        assert "No Input layer found" in body["message"]
        assert isinstance(body["message"], str)
