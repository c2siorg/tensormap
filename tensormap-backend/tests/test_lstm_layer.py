"""Tests for LSTM layer support in model_generation._build_layer."""

import pytest
import tensorflow as tf

from app.services.model_generation import _build_layer


def _make_node(units, return_sequences="false"):
    return {
        "id": "lstm-test",
        "type": "customlstm",
        "data": {"params": {"units": units, "returnSequences": return_sequences}},
    }


def test_build_lstm_layer_basic():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node("64")
    output = _build_layer(node, input_tensor)
    assert output.shape[-1] == 64


def test_build_lstm_return_sequences_false():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node("32", "false")
    output = _build_layer(node, input_tensor)
    assert len(output.shape) == 2


def test_build_lstm_return_sequences_true():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node("32", "true")
    output = _build_layer(node, input_tensor)
    assert len(output.shape) == 3


def test_build_lstm_invalid_units_empty():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node("")
    with pytest.raises(ValueError, match="Invalid LSTM units"):
        _build_layer(node, input_tensor)


def test_build_lstm_invalid_units_zero():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node(0)
    with pytest.raises(ValueError, match="LSTM units must be a positive integer"):
        _build_layer(node, input_tensor)


def test_build_lstm_return_sequences_true_shape():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node("32", "true")
    output = _build_layer(node, input_tensor)
    # return_sequences=True preserves the time dimension
    assert len(output.shape) == 3
    assert output.shape[-1] == 32


def test_build_lstm_invalid_units_negative():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node(-10)
    with pytest.raises(ValueError, match="LSTM units must be a positive integer"):
        _build_layer(node, input_tensor)


def test_build_lstm_invalid_units_nan():
    input_tensor = tf.keras.Input(shape=(10, 8))
    node = _make_node("NaN")
    with pytest.raises(ValueError):
        _build_layer(node, input_tensor)
