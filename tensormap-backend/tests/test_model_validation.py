"""Unit tests for model validation using pytest."""

import pytest

from app.services.model_generation import model_generation


class TestModelValidation:
    """Test suite for model_generation function."""

    def test_happy_path_input_to_dense(self):
        """Test that a simple valid graph (Input -> Dense) returns success without raising ValueError."""
        # Create a simple valid model: Input layer connected to Dense layer
        model_params = {
            "nodes": [
                {
                    "id": "input-1",
                    "type": "custominput",
                    "data": {
                        "params": {
                            "dim-1": 10,
                            "dim-2": 0,
                            "dim-3": 0,
                        }
                    },
                },
                {
                    "id": "dense-1",
                    "type": "customdense",
                    "data": {
                        "params": {
                            "units": 5,
                            "activation": "relu",
                        }
                    },
                },
            ],
            "edges": [
                {
                    "source": "input-1",
                    "target": "dense-1",
                }
            ],
        }

        # This should not raise any ValueError
        result = model_generation(model_params)

        # Verify the result is a valid Keras JSON model
        assert isinstance(result, dict)
        assert "config" in result or "class_name" in result
        assert result.get("class_name") == "Model"

    def test_happy_path_single_input_output(self):
        """Test a single input and output layer succeeds."""
        model_params = {
            "nodes": [
                {
                    "id": "input-node",
                    "type": "custominput",
                    "data": {
                        "params": {
                            "dim-1": 28,
                            "dim-2": 28,
                            "dim-3": 1,
                        }
                    },
                },
            ],
            "edges": [],
        }

        result = model_generation(model_params)

        # Should have input and output layers
        assert isinstance(result, dict)
        assert "config" in result or "class_name" in result

    def test_disconnected_node_raises_error(self):
        """Test that a disconnected node raises ValueError with appropriate message."""
        # Create a model with a disconnected node
        model_params = {
            "nodes": [
                {
                    "id": "input-1",
                    "type": "custominput",
                    "data": {
                        "params": {
                            "dim-1": 10,
                            "dim-2": 0,
                            "dim-3": 0,
                        }
                    },
                },
                {
                    "id": "dense-1",
                    "type": "customdense",
                    "data": {
                        "params": {
                            "units": 5,
                            "activation": "relu",
                        }
                    },
                },
                {
                    "id": "disconnected-dense",
                    "type": "customdense",
                    "data": {
                        "params": {
                            "units": 3,
                            "activation": "sigmoid",
                        }
                    },
                },
            ],
            "edges": [
                {
                    "source": "input-1",
                    "target": "dense-1",
                }
                # Note: disconnected-dense is not connected to anything
            ],
        }

        # This should raise ValueError for disconnected node
        with pytest.raises(ValueError) as exc_info:
            model_generation(model_params)

        assert "Disconnected graph" in str(exc_info.value)
        assert "node(s) are not connected" in str(exc_info.value)

    def test_no_input_layer_raises_error(self):
        """Test that missing input layer raises ValueError."""
        # Model without any input nodes
        model_params = {
            "nodes": [
                {
                    "id": "dense-1",
                    "type": "customdense",
                    "data": {
                        "params": {
                            "units": 5,
                            "activation": "relu",
                        }
                    },
                },
            ],
            "edges": [],
        }

        with pytest.raises(ValueError) as exc_info:
            model_generation(model_params)

        assert "No Input layer found" in str(exc_info.value)

    def test_invalid_units_parameter(self):
        """Test that invalid units parameter raises ValueError."""
        model_params = {
            "nodes": [
                {
                    "id": "input-1",
                    "type": "custominput",
                    "data": {
                        "params": {
                            "dim-1": 10,
                            "dim-2": 0,
                            "dim-3": 0,
                        }
                    },
                },
                {
                    "id": "dense-1",
                    "type": "customdense",
                    "data": {
                        "params": {
                            "units": "not-a-number",  # Invalid units
                            "activation": "relu",
                        }
                    },
                },
            ],
            "edges": [
                {
                    "source": "input-1",
                    "target": "dense-1",
                }
            ],
        }

        with pytest.raises(ValueError) as exc_info:
            model_generation(model_params)

        assert "Invalid 'units' parameter" in str(exc_info.value)

    def test_float_units_parameter_raises_error(self):
        """Test that float units are rejected to avoid silent truncation."""
        model_params = {
            "nodes": [
                {
                    "id": "input-1",
                    "type": "custominput",
                    "data": {
                        "params": {
                            "dim-1": 10,
                            "dim-2": 0,
                            "dim-3": 0,
                        }
                    },
                },
                {
                    "id": "dense-1",
                    "type": "customdense",
                    "data": {
                        "params": {
                            "units": 10.5,
                            "activation": "relu",
                        }
                    },
                },
            ],
            "edges": [
                {
                    "source": "input-1",
                    "target": "dense-1",
                }
            ],
        }

        with pytest.raises(ValueError) as exc_info:
            model_generation(model_params)

        assert "Invalid 'units' parameter" in str(exc_info.value)

    def test_unknown_node_type_raises_error(self):
        """Test that unknown node type raises ValueError."""
        model_params = {
            "nodes": [
                {
                    "id": "input-1",
                    "type": "custominput",
                    "data": {
                        "params": {
                            "dim-1": 10,
                            "dim-2": 0,
                            "dim-3": 0,
                        }
                    },
                },
                {
                    "id": "unknown-1",
                    "type": "unknowntype",  # Unknown node type
                    "data": {"params": {}},
                },
            ],
            "edges": [
                {
                    "source": "input-1",
                    "target": "unknown-1",
                }
            ],
        }

        with pytest.raises(ValueError) as exc_info:
            model_generation(model_params)

        assert "Unknown node type" in str(exc_info.value)
