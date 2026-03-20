"""
Unit and integration tests for model_generation and the layer registry.
"""

import json

import pytest
import tensorflow as tf

from app.layers.registry import build_layer as _build_layer
from app.layers.registry import get_layer_schema
from app.services.model_generation import model_generation


def _input_node(node_id, dims):
    params = {f"dim-{i + 1}": d for i, d in enumerate(dims)}
    return {"id": node_id, "type": "custominput", "data": {"params": params}}


def _dense_node(node_id, units=32, activation="relu"):
    return {"id": node_id, "type": "customdense", "data": {"params": {"units": units, "activation": activation}}}


def _flatten_node(node_id):
    return {"id": node_id, "type": "customflatten", "data": {"params": {}}}


def _conv_node(node_id, filters=16, kernel=(3, 3), stride=(1, 1), padding="valid", activation="relu"):
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


def _maxpool_node(node_id, pool=2, stride=2):
    return {
        "id": node_id,
        "type": "custommaxpool2d",
        "data": {
            "params": {
                "poolX": pool,
                "poolY": pool,
                "strideX": stride,
                "strideY": stride,
                "padding": "valid",
            }
        },
    }


def _dropout_node(node_id, rate=0.5):
    return {"id": node_id, "type": "customdropout", "data": {"params": {"rate": rate}}}


def _batchnorm_node(node_id):
    return {"id": node_id, "type": "custombatchnorm", "data": {"params": {"momentum": 0.99, "epsilon": 0.001}}}


def _lstm_node(node_id, units=32, return_sequences=False):
    return {
        "id": node_id,
        "type": "customlstm",
        "data": {
            "params": {
                "units": units,
                "return_sequences": str(return_sequences).lower(),
                "dropout": 0.0,
            }
        },
    }


def _gru_node(node_id, units=32, return_sequences=False):
    return {
        "id": node_id,
        "type": "customgru",
        "data": {
            "params": {
                "units": units,
                "return_sequences": str(return_sequences).lower(),
                "dropout": 0.0,
            }
        },
    }


def _edge(source, target):
    return {"source": source, "target": target}


class TestBuildLayer:
    def test_dense_output_shape(self):
        inp = tf.keras.Input(shape=(10,), name="i")
        assert _build_layer(_dense_node("d1", 64), inp).shape == (None, 64)

    def test_dense_sigmoid(self):
        inp = tf.keras.Input(shape=(5,), name="i")
        assert _build_layer(_dense_node("d1", 1, "sigmoid"), inp).shape == (None, 1)

    def test_flatten_output_shape(self):
        inp = tf.keras.Input(shape=(4, 4, 3), name="i")
        assert _build_layer(_flatten_node("f1"), inp).shape == (None, 48)

    def test_conv2d_valid_padding(self):
        inp = tf.keras.Input(shape=(28, 28, 1), name="i")
        assert _build_layer(_conv_node("c1", 16, (3, 3), (1, 1), "valid"), inp).shape == (None, 26, 26, 16)

    def test_conv2d_same_padding(self):
        inp = tf.keras.Input(shape=(16, 16, 1), name="i")
        assert _build_layer(_conv_node("c1", 8, (3, 3), (1, 1), "same"), inp).shape == (None, 16, 16, 8)

    def test_conv2d_activation_none(self):
        inp = tf.keras.Input(shape=(8, 8, 1), name="i")
        assert _build_layer(_conv_node("c1", activation="none"), inp) is not None

    def test_unknown_type_raises(self):
        inp = tf.keras.Input(shape=(5,), name="i")
        with pytest.raises(ValueError, match="Unknown node type"):
            _build_layer({"id": "x", "type": "custom_unknown", "data": {"params": {}}}, inp)

    def test_maxpooling2d(self):
        inp = tf.keras.Input(shape=(28, 28, 16), name="i")
        assert _build_layer(_maxpool_node("mp1"), inp).shape == (None, 14, 14, 16)

    def test_dropout_passthrough(self):
        inp = tf.keras.Input(shape=(64,), name="i")
        assert _build_layer(_dropout_node("do1", 0.3), inp).shape == (None, 64)

    def test_batchnorm_passthrough(self):
        inp = tf.keras.Input(shape=(32,), name="i")
        assert _build_layer(_batchnorm_node("bn1"), inp).shape == (None, 32)

    def test_lstm_output_shape(self):
        inp = tf.keras.Input(shape=(10, 8), name="i")
        assert _build_layer(_lstm_node("l1", 32, False), inp).shape == (None, 32)

    def test_lstm_return_sequences(self):
        inp = tf.keras.Input(shape=(10, 8), name="i")
        assert _build_layer(_lstm_node("l1", 32, True), inp).shape == (None, 10, 32)

    def test_gru_output_shape(self):
        inp = tf.keras.Input(shape=(10, 8), name="i")
        assert _build_layer(_gru_node("g1", 16), inp).shape == (None, 16)


class TestModelGeneration:
    def test_returns_dict(self):
        p = {"nodes": [_input_node("i", [10]), _dense_node("o", 1, "linear")], "edges": [_edge("i", "o")]}
        assert isinstance(model_generation(p), dict)

    def test_input_dense_shapes(self):
        p = {"nodes": [_input_node("i", [10]), _dense_node("o", 1, "linear")], "edges": [_edge("i", "o")]}
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.input_shape == (None, 10) and m.output_shape == (None, 1)

    def test_multi_layer_chain(self):
        p = {
            "nodes": [
                _input_node("x", [20]),
                _dense_node("h1", 64),
                _dense_node("h2", 32),
                _dense_node("o", 1, "sigmoid"),
            ],
            "edges": [_edge("x", "h1"), _edge("h1", "h2"), _edge("h2", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.output_shape == (None, 1) and len(m.layers) == 4

    def test_conv_flatten_dense(self):
        p = {
            "nodes": [
                _input_node("img", [28, 28, 1]),
                _conv_node("c1", 16, (3, 3), (1, 1), "valid"),
                _flatten_node("fl"),
                _dense_node("o", 10, "softmax"),
            ],
            "edges": [_edge("img", "c1"), _edge("c1", "fl"), _edge("fl", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.output_shape == (None, 10)

    def test_conv_maxpool_flatten_dense(self):
        p = {
            "nodes": [
                _input_node("img", [28, 28, 1]),
                _conv_node("c1", 16, (3, 3), (1, 1), "same"),
                _maxpool_node("mp1"),
                _flatten_node("fl"),
                _dense_node("o", 10, "softmax"),
            ],
            "edges": [_edge("img", "c1"), _edge("c1", "mp1"), _edge("mp1", "fl"), _edge("fl", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.output_shape == (None, 10)

    def test_dense_dropout_batchnorm(self):
        p = {
            "nodes": [
                _input_node("x", [64]),
                _dense_node("h1", 64),
                _dropout_node("do1", 0.3),
                _batchnorm_node("bn1"),
                _dense_node("o", 1, "sigmoid"),
            ],
            "edges": [_edge("x", "h1"), _edge("h1", "do1"), _edge("do1", "bn1"), _edge("bn1", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.output_shape == (None, 1)

    def test_lstm_dense(self):
        p = {
            "nodes": [_input_node("seq", [10, 8]), _lstm_node("l1", 32, False), _dense_node("o", 1, "sigmoid")],
            "edges": [_edge("seq", "l1"), _edge("l1", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.output_shape == (None, 1)

    def test_stacked_lstm(self):
        p = {
            "nodes": [
                _input_node("seq", [10, 8]),
                _lstm_node("l1", 32, True),
                _lstm_node("l2", 16, False),
                _dense_node("o", 1, "sigmoid"),
            ],
            "edges": [_edge("seq", "l1"), _edge("l1", "l2"), _edge("l2", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.output_shape == (None, 1)

    def test_multi_input(self):
        p = {
            "nodes": [_input_node("i1", [5]), _input_node("i2", [5]), _dense_node("o", 1, "sigmoid")],
            "edges": [_edge("i1", "o"), _edge("i2", "o")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert len(m.inputs) == 2

    def test_multiple_outputs(self):
        p = {
            "nodes": [_input_node("x", [10]), _dense_node("o1", 1, "sigmoid"), _dense_node("o2", 5, "softmax")],
            "edges": [_edge("x", "o1"), _edge("x", "o2")],
        }
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert len(m.outputs) == 2

    def test_single_input_no_edges_raises(self):
        with pytest.raises(ValueError):
            model_generation({"nodes": [_input_node("solo", [4])], "edges": []})

    def test_unknown_layer_raises(self):
        p = {
            "nodes": [_input_node("x", [5]), {"id": "bad", "type": "customunknown", "data": {"params": {}}}],
            "edges": [_edge("x", "bad")],
        }
        with pytest.raises(ValueError, match="Unknown node type"):
            model_generation(p)

    def test_json_serialisable(self):
        p = {"nodes": [_input_node("x", [8]), _dense_node("y", 4, "relu")], "edges": [_edge("x", "y")]}
        s = json.dumps(model_generation(p))
        assert len(s) > 0

    def test_3d_input_shape(self):
        p = {"nodes": [_input_node("img", [28, 28, 1]), _dense_node("o", 10, "softmax")], "edges": [_edge("img", "o")]}
        m = tf.keras.models.model_from_json(json.dumps(model_generation(p)))
        assert m.input_shape == (None, 28, 28, 1)


class TestLayerSchema:
    def test_returns_list_of_15_plus(self):
        assert len(get_layer_schema()) >= 15

    def test_required_fields(self):
        required = {"node_type", "display_name", "default_params", "description", "category"}
        for entry in get_layer_schema():
            assert required <= entry.keys()

    def test_all_categories_present(self):
        cats = {e["category"] for e in get_layer_schema()}
        assert {"core", "pooling", "regularization", "recurrent"} <= cats

    def test_original_four_layers_present(self):
        types = {e["node_type"] for e in get_layer_schema()}
        assert {"custominput", "customdense", "customflatten", "customconv"} <= types

    def test_new_layers_present(self):
        types = {e["node_type"] for e in get_layer_schema()}
        assert {
            "custommaxpool2d",
            "customavgpool2d",
            "customdropout",
            "custombatchnorm",
            "customlayernorm",
            "customlstm",
            "customgru",
            "customsimplernn",
            "customreshape",
            "customembedding",
        } <= types
