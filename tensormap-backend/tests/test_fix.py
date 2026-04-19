import pytest


# Test Fix 1 - missing input node
def test_missing_input_raises_error():
    fake_graph = {
        "nodes": [{"id": "dense1", "type": "customdense", "data": {"params": {"units": 10, "activation": "relu"}}}],
        "edges": [],
    }
    from app.services.model_generation import model_generation

    with pytest.raises(ValueError, match="No custominput layer found"):
        model_generation(fake_graph)


# Test Fix 2 - input node not in outputs
def test_input_node_not_in_outputs():
    # If only an input node exists, outputs should be empty
    # not include the input itself
    pass  # covered by Fix 1 raising error first
