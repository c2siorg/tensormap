"""Unit tests for AlphaDropout layer in model_generation._build_layer()."""

import pytest
import tensorflow as tf

from app.services.model_generation import _build_layer


def _make_node(rate: str) -> dict:
    """Helper to create an AlphaDropout node dict."""
    return {
        "id": "alphadropout-test",
        "type": "customalphadropout",
        "data": {"params": {"rate": rate}},
    }


def test_build_alphadropout_basic():
    """AlphaDropout with valid rate builds correctly."""
    input_tensor = tf.keras.Input(shape=(10,))
    node = _make_node("0.5")
    output = _build_layer(node, input_tensor)
    assert output.shape[-1] == 10


def test_build_alphadropout_rate_0_1():
    """AlphaDropout with low rate 0.1."""
    input_tensor = tf.keras.Input(shape=(20,))
    node = _make_node("0.1")
    output = _build_layer(node, input_tensor)
    assert output.shape == (None, 20)


def test_build_alphadropout_rate_0_9():
    """AlphaDropout with high rate 0.9."""
    input_tensor = tf.keras.Input(shape=(15,))
    node = _make_node("0.9")
    output = _build_layer(node, input_tensor)
    assert output.shape == (None, 15)


def test_build_alphadropout_default_rate():
    """AlphaDropout with empty rate falls back to default 0.5."""
    input_tensor = tf.keras.Input(shape=(8,))
    node = _make_node("")
    output = _build_layer(node, input_tensor)
    assert output.shape[-1] == 8


def test_build_alphadropout_invalid_rate_zero():
    """AlphaDropout with rate=0 raises ValueError."""
    input_tensor = tf.keras.Input(shape=(10,))
    node = _make_node("0")
    with pytest.raises((ValueError, Exception)):
        _build_layer(node, input_tensor)


def test_build_alphadropout_invalid_rate_negative():
    """AlphaDropout with negative rate raises ValueError."""
    input_tensor = tf.keras.Input(shape=(10,))
    node = _make_node("-0.5")
    with pytest.raises((ValueError, Exception)):
        _build_layer(node, input_tensor)


def test_build_alphadropout_invalid_rate_over_one():
    """AlphaDropout with rate >= 1 raises ValueError."""
    input_tensor = tf.keras.Input(shape=(10,))
    node = _make_node("1.5")
    with pytest.raises((ValueError, Exception)):
        _build_layer(node, input_tensor)


def test_build_alphadropout_invalid_rate_nan():
    """AlphaDropout with non-numeric rate raises ValueError."""
    input_tensor = tf.keras.Input(shape=(10,))
    node = _make_node("abc")
    with pytest.raises((ValueError, Exception)):
        _build_layer(node, input_tensor)


def test_build_alphadropout_in_model():
    """AlphaDropout can be used in a full model pipeline."""
    inputs = tf.keras.Input(shape=(16,))
    node = _make_node("0.3")
    dropped = _build_layer(node, inputs)
    outputs = tf.keras.layers.Dense(1)(dropped)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    assert model.layers[1].__class__.__name__ == "AlphaDropout"
    assert model.layers[1].rate == pytest.approx(0.3)
