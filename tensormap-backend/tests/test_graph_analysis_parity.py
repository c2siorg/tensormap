"""The analyzer must agree with Keras.

Each graph is analysed statically and then built with the real ``TensorFlowGenerator``. Output
shapes and parameter counts have to match layer by layer, and a graph the analyzer rejects as
invalid has to fail to build in Keras too. This is what keeps the pure-Python rules honest.
"""

import pytest

from app.generators.tensorflow_generator import TensorFlowGenerator, TensorFlowGeneratorError
from app.ir.analysis import analyze_graph
from tests.test_graph_analysis import chain, image_cnn, inp, make_graph

GRAPHS = {
    "dense": lambda: chain(
        inp(10),
        ("d1", {"layer_type": "dense", "units": 16, "activation": "relu"}),
        ("d2", {"layer_type": "dense", "units": 3, "activation": "softmax"}),
    ),
    "cnn": image_cnn,
    "cnn_same_stride2": lambda: chain(
        inp(49),
        ("rs", {"layer_type": "reshape", "target_shape": "7,7,1"}),
        ("c", {"layer_type": "conv2d", "filters": 5, "kernel_size": 3, "strides": 2, "padding": "same"}),
        ("ap", {"layer_type": "avgpool2d", "pool_size": 2, "strides": 2, "padding": "same"}),
        ("g", {"layer_type": "globalavgpool2d"}),
    ),
    "cnn_batchnorm_dropout": lambda: chain(
        inp(64),
        ("rs", {"layer_type": "reshape", "target_shape": "8,8,1"}),
        ("c", {"layer_type": "conv2d", "filters": 4, "kernel_size": 3, "padding": "valid"}),
        ("bn", {"layer_type": "batchnorm"}),
        ("dr", {"layer_type": "dropout", "rate": 0.3}),
        ("fl", {"layer_type": "flatten"}),
        ("d", {"layer_type": "dense", "units": 2}),
    ),
    "lstm": lambda: chain(
        inp(20),
        ("rs", {"layer_type": "reshape", "target_shape": "5,4"}),
        ("l1", {"layer_type": "lstm", "units": 8, "return_sequences": True}),
        ("l2", {"layer_type": "lstm", "units": 6}),
        ("d", {"layer_type": "dense", "units": 1}),
    ),
    "gru": lambda: chain(
        inp(20),
        ("rs", {"layer_type": "reshape", "target_shape": "-1,2"}),
        ("g1", {"layer_type": "gru", "units": 7, "return_sequences": True}),
        ("g2", {"layer_type": "gru", "units": 5}),
    ),
    "simplernn": lambda: chain(
        inp(12),
        ("rs", {"layer_type": "reshape", "target_shape": "4,3"}),
        ("r", {"layer_type": "simplernn", "units": 9}),
    ),
    "embedding_lstm": lambda: chain(
        inp(30),
        ("e", {"layer_type": "embedding", "input_dim": 500, "output_dim": 12}),
        ("l", {"layer_type": "lstm", "units": 8}),
        ("d", {"layer_type": "dense", "units": 1, "activation": "sigmoid"}),
    ),
    "dense_after_conv_keeps_rank": lambda: chain(
        inp(16),
        ("rs", {"layer_type": "reshape", "target_shape": "4,4,1"}),
        ("c", {"layer_type": "conv2d", "filters": 2, "padding": "same"}),
        ("d", {"layer_type": "dense", "units": 3}),
    ),
    "concatenate_branches": lambda: make_graph(
        [
            inp(8),
            ("a", {"layer_type": "dense", "units": 5}),
            ("b", {"layer_type": "dense", "units": 7}),
            ("cat", {"layer_type": "concatenate", "axis": -1}),
            ("out", {"layer_type": "dense", "units": 2}),
        ],
        [("in", "a"), ("in", "b"), ("a", "cat"), ("b", "cat"), ("cat", "out")],
    ),
    "concatenate_positive_axis": lambda: make_graph(
        [
            inp(12),
            ("r1", {"layer_type": "reshape", "target_shape": "3,4"}),
            ("r2", {"layer_type": "reshape", "target_shape": "3,4"}),
            ("cat", {"layer_type": "concatenate", "axis": 1}),
        ],
        [("in", "r1"), ("in", "r2"), ("r1", "cat"), ("r2", "cat")],
    ),
    "implicit_merge": lambda: make_graph(
        [
            inp(8),
            ("a", {"layer_type": "dense", "units": 5}),
            ("b", {"layer_type": "dense", "units": 7}),
            ("out", {"layer_type": "dense", "units": 2}),
        ],
        [("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")],
    ),
    "two_inputs": lambda: make_graph(
        [
            ("i1", {"layer_type": "input", "shape": 4}),
            ("i2", {"layer_type": "input", "shape": 6}),
            ("cat", {"layer_type": "concatenate", "axis": -1}),
            ("out", {"layer_type": "dense", "units": 3}),
        ],
        [("i1", "cat"), ("i2", "cat"), ("cat", "out")],
    ),
}

# Graphs the analyzer must reject, and which Keras must fail to build too.
INVALID_GRAPHS = {
    "conv_on_flat_input": lambda: chain(inp(784), ("c", {"layer_type": "conv2d", "filters": 4})),
    "lstm_on_flat_input": lambda: chain(inp(16), ("l", {"layer_type": "lstm", "units": 4})),
    "kernel_larger_than_input": lambda: chain(
        inp(9),
        ("rs", {"layer_type": "reshape", "target_shape": "3,3,1"}),
        ("c", {"layer_type": "conv2d", "filters": 2, "kernel_size": 5, "padding": "valid"}),
    ),
    "reshape_size_mismatch": lambda: chain(inp(10), ("rs", {"layer_type": "reshape", "target_shape": "3,3"})),
    "concatenate_mismatch": lambda: make_graph(
        [
            inp(8),
            ("r1", {"layer_type": "reshape", "target_shape": "2,4"}),
            ("r2", {"layer_type": "reshape", "target_shape": "4,2"}),
            ("cat", {"layer_type": "concatenate", "axis": -1}),
        ],
        [("in", "r1"), ("in", "r2"), ("r1", "cat"), ("r2", "cat")],
    ),
}


@pytest.mark.parametrize("name", GRAPHS)
def test_shapes_and_params_match_keras(name):
    graph = GRAPHS[name]()
    analysis = analyze_graph(graph)
    assert analysis.ok and analysis.complete, analysis.diagnostics

    model = TensorFlowGenerator().build_model(graph)

    for node in analysis.nodes:
        if node.layer_type == "input":
            continue
        layer = model.get_layer(node.node_id)
        expected = tuple(layer.output.shape)[1:]  # Drop the batch axis.
        assert node.output_shape == expected, f"{node.node_id}: {node.output_shape} != {expected}"
        assert node.params == layer.count_params(), f"{node.node_id} params"
        trainable = sum(int(w.numpy().size) for w in layer.trainable_weights)
        assert node.trainable_params == trainable, f"{node.node_id} trainable params"

    assert analysis.total_params == model.count_params()
    assert analysis.trainable_params == sum(int(w.numpy().size) for w in model.trainable_weights)
    assert sorted(analysis.output_node_ids) == sorted(t._keras_history.operation.name for t in model.outputs)


@pytest.mark.parametrize("name", INVALID_GRAPHS)
def test_graphs_rejected_by_the_analyzer_also_fail_in_keras(name):
    graph = INVALID_GRAPHS[name]()
    assert not analyze_graph(graph).ok
    with pytest.raises((TensorFlowGeneratorError, ValueError)):
        TensorFlowGenerator().build_model(graph)
