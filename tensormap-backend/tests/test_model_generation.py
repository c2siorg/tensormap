"""
Unit and integration tests for the model_generation service.

Covers:
  - _build_layer() for each supported node type (Dense, Flatten, Conv2D)
  - _build_layer() error handling for unknown node types
  - model_generation() end-to-end: linear, branching, multi-input, conv
  - Integration: generated JSON round-trips through tf.keras.models.model_from_json
"""

import json

import pytest
import tensorflow as tf

from app.services.model_generation import _build_layer, model_generation




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

    def test_unknown_node_type_raises_value_error(self):
        input_t = tf.keras.Input(shape=(5,), name="inp")
        node = {"id": "x", "type": "custom_unknown", "data": {"params": {}}}
        with pytest.raises(ValueError, match="Unknown node type"):
            _build_layer(node, input_t)


# ===================================================================
# Tests for model_generation()
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

    def test_single_input_no_edges(self):
        """A graph with only an input node and no edges cannot form a valid Keras
        model — the input tensor is also the output tensor, which Keras 3's
        Functional API rejects with ValueError."""
        params = {
            "nodes": [_input_node("solo", [4])],
            "edges": [],
        }
        with pytest.raises(ValueError):
            model_generation(params)

    def test_unknown_layer_type_raises_value_error(self):
        """An unsupported node type in the graph must propagate a ValueError."""
        params = {
            "nodes": [
                _input_node("x", [5]),
                {"id": "bad", "type": "customunknown", "data": {"params": {}}},
            ],
            "edges": [_edge("x", "bad")],
        }
        with pytest.raises(ValueError, match="Unknown node type"):
            model_generation(params)

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

