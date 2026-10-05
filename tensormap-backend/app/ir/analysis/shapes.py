"""Shape, parameter-count and cost rules for every layer in the registry.

Each rule is a pure function of a layer's validated parameters and the shapes feeding it. No
TensorFlow is imported, so analysing a graph takes microseconds and can run on every canvas edit.
``tests/test_graph_analysis_parity.py`` checks every rule against a real Keras model, and
``test_every_registry_layer_has_a_rule`` fails when a layer is added to the registry without one.

Shapes exclude the batch axis. Padding and output-size arithmetic follows Keras.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from app.ir.analysis.models import DiagnosticCode, Severity, Shape
from app.ir.schema import (
    AvgPool2DParams,
    BatchNormParams,
    ConcatenateParams,
    Conv2DParams,
    DenseParams,
    DropoutParams,
    EmbeddingParams,
    FlattenParams,
    GlobalAvgPool2DParams,
    GRUParams,
    InputParams,
    LSTMParams,
    MaxPool2DParams,
    ReshapeParams,
    SimpleRNNParams,
)


class ShapeError(Exception):
    """A layer cannot accept the shapes it was given."""

    def __init__(
        self,
        code: DiagnosticCode,
        detail: str,
        suggestion: str | None = None,
        severity: Severity = Severity.ERROR,
    ):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.suggestion = suggestion
        self.severity = severity


@dataclass(frozen=True)
class LayerResult:
    """Output of one rule."""

    output_shape: Shape
    params: int = 0
    non_trainable: int = 0  # Part of ``params`` that the optimiser does not update.
    macs: int = 0
    warning: ShapeError | None = None  # Non-blocking finding; the shape is still valid.


Rule = Callable[..., LayerResult]
_RULES: dict[str, Rule] = {}


def shape_rule(layer_type: str) -> Callable[[Rule], Rule]:
    """Register the rule for a registry layer type."""

    def register(fn: Rule) -> Rule:
        _RULES[layer_type] = fn
        return fn

    return register


def get_rule(layer_type: str) -> Rule | None:
    return _RULES.get(layer_type)


def registered_layer_types() -> set[str]:
    return set(_RULES)


def format_shape(shape: Shape) -> str:
    """``(28, 28, 1)`` for display; one-element shapes keep the Keras trailing comma."""
    return "(" + ", ".join(str(d) for d in shape) + ("," if len(shape) == 1 else "") + ")"


def _prod(dims: Shape) -> int:
    return math.prod(dims)


def _describe_rank(shape: Shape) -> str:
    return f"{len(shape)}-D input {format_shape(shape)}"


# ----------------------------------------------------------------------------
# Multiple inputs
# ----------------------------------------------------------------------------


def concat_shapes(shapes: list[Shape], axis: int = -1) -> Shape:
    """Concatenate shapes the way Keras ``Concatenate(axis=axis)`` would.

    ``axis`` follows Keras and counts the batch axis, so ``-1`` is the last feature axis, ``1``
    is the first and ``0`` (the batch itself) is rejected.
    """
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes[1:]):
        raise ShapeError(
            DiagnosticCode.SHAPE_MISMATCH,
            "cannot merge inputs of different ranks: " + ", ".join(format_shape(s) for s in shapes) + ".",
            "Reshape or Flatten the inputs so they have the same number of dimensions.",
        )
    full_rank = rank + 1  # Keras counts the batch axis.
    if not -full_rank <= axis < full_rank:
        raise ShapeError(DiagnosticCode.SHAPE_MISMATCH, f"axis {axis} is out of range for {rank}-D inputs.")
    full_axis = axis if axis >= 0 else axis + full_rank
    if full_axis == 0:
        raise ShapeError(DiagnosticCode.SHAPE_MISMATCH, "cannot concatenate along the batch axis (axis 0).")
    ax = full_axis - 1
    for s in shapes[1:]:
        if any(a != b for i, (a, b) in enumerate(zip(shapes[0], s, strict=True)) if i != ax):
            raise ShapeError(
                DiagnosticCode.SHAPE_MISMATCH,
                "inputs differ in a dimension other than the merge axis: "
                + ", ".join(format_shape(x) for x in shapes)
                + ".",
                "Every dimension except the one being concatenated must match.",
            )
    merged = list(shapes[0])
    merged[ax] = sum(s[ax] for s in shapes)
    return tuple(merged)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _spatial_out(size: int, kernel: int, stride: int, padding: str) -> int:
    if padding == "same":
        return math.ceil(size / stride)
    return (size - kernel) // stride + 1 if size >= kernel else 0


def _need_rank(shape: Shape, rank: int, meaning: str, suggestion: str | None) -> None:
    if len(shape) != rank:
        raise ShapeError(
            DiagnosticCode.RANK_MISMATCH,
            f"expects a {rank}-D input {meaning} but receives a {_describe_rank(shape)}.",
            suggestion,
        )


def _image_hint(shape: Shape) -> str:
    """Suggest a Reshape that would turn ``shape`` into an image."""
    if len(shape) == 1:
        n = shape[0]
        side = math.isqrt(n)
        if side * side == n:
            return f"Insert a Reshape layer with target shape '{side},{side},1' before this layer."
        return f"Insert a Reshape layer to (height, width, channels) whose sizes multiply to {n} before this layer."
    if len(shape) == 2:
        return f"Insert a Reshape layer with target shape '{shape[0]},{shape[1]},1' before this layer."
    return "Insert a Reshape layer to (height, width, channels) before this layer."


def _sequence_hint(shape: Shape) -> str:
    if len(shape) == 1:
        return (
            f"Insert a Reshape layer with target shape '{shape[0]},1' (timesteps, features) before this layer, "
            "or use an Embedding layer if the input holds token ids."
        )
    if len(shape) == 3:
        h, w, c = shape
        return f"Insert a Reshape layer with target shape '{h * w},{c}' (timesteps, features) before this layer."
    return "Insert a Reshape layer to (timesteps, features) before this layer."


# ----------------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------------


@shape_rule("input")
def _input(p: InputParams, inputs: list[Shape]) -> LayerResult:
    return LayerResult((p.shape,))


@shape_rule("dense")
def _dense(p: DenseParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    features = shape[-1]
    lead = shape[:-1]
    warning = None
    if len(shape) >= 3:
        warning = ShapeError(
            DiagnosticCode.UNFLATTENED_DENSE,
            f"Dense is applied to every position of a {_describe_rank(shape)} separately, "
            f"so its output stays {len(shape)}-D.",
            "Add a Flatten layer before Dense if you want one prediction per sample.",
            Severity.WARNING,
        )
    return LayerResult(
        (*lead, p.units),
        params=features * p.units + p.units,
        macs=_prod(lead) * features * p.units,
        warning=warning,
    )


@shape_rule("flatten")
def _flatten(p: FlattenParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    return LayerResult((_prod(shape),))


@shape_rule("conv2d")
def _conv2d(p: Conv2DParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    _need_rank(shape, 3, "(height, width, channels)", _image_hint(shape))
    h, w, c = shape
    oh = _spatial_out(h, p.kernel_size, p.strides, p.padding)
    ow = _spatial_out(w, p.kernel_size, p.strides, p.padding)
    if oh < 1 or ow < 1:
        raise ShapeError(
            DiagnosticCode.INVALID_OUTPUT_SIZE,
            f"a {p.kernel_size}x{p.kernel_size} kernel with 'valid' padding does not fit in a {h}x{w} input.",
            "Use a smaller kernel size, or set padding to 'same'.",
        )
    return LayerResult(
        (oh, ow, p.filters),
        params=p.kernel_size * p.kernel_size * c * p.filters + p.filters,
        macs=oh * ow * p.kernel_size * p.kernel_size * c * p.filters,
    )


def _pool2d(p: MaxPool2DParams | AvgPool2DParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    _need_rank(shape, 3, "(height, width, channels)", _image_hint(shape))
    h, w, c = shape
    oh = _spatial_out(h, p.pool_size, p.strides, p.padding)
    ow = _spatial_out(w, p.pool_size, p.strides, p.padding)
    if oh < 1 or ow < 1:
        raise ShapeError(
            DiagnosticCode.INVALID_OUTPUT_SIZE,
            f"a {p.pool_size}x{p.pool_size} pooling window with 'valid' padding does not fit in a {h}x{w} input.",
            "Use a smaller pool size, or set padding to 'same'.",
        )
    return LayerResult((oh, ow, c))


@shape_rule("maxpool2d")
def _maxpool2d(p: MaxPool2DParams, inputs: list[Shape]) -> LayerResult:
    return _pool2d(p, inputs)


@shape_rule("avgpool2d")
def _avgpool2d(p: AvgPool2DParams, inputs: list[Shape]) -> LayerResult:
    return _pool2d(p, inputs)


@shape_rule("globalavgpool2d")
def _globalavgpool2d(p: GlobalAvgPool2DParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    _need_rank(shape, 3, "(height, width, channels)", _image_hint(shape))
    return LayerResult((shape[2],))


def _recurrent(
    p: LSTMParams | GRUParams | SimpleRNNParams,
    inputs: list[Shape],
    gates: int,
    bias_rows: int,
) -> LayerResult:
    (shape,) = inputs
    _need_rank(shape, 2, "(timesteps, features)", _sequence_hint(shape))
    steps, features = shape
    u = p.units
    out: Shape = (steps, u) if p.return_sequences else (u,)
    return LayerResult(
        out,
        params=gates * (features * u + u * u + bias_rows * u),
        macs=steps * gates * (features * u + u * u),
    )


@shape_rule("lstm")
def _lstm(p: LSTMParams, inputs: list[Shape]) -> LayerResult:
    return _recurrent(p, inputs, gates=4, bias_rows=1)


@shape_rule("gru")
def _gru(p: GRUParams, inputs: list[Shape]) -> LayerResult:
    # Keras GRU defaults to reset_after=True, which keeps separate input and recurrent biases.
    return _recurrent(p, inputs, gates=3, bias_rows=2)


@shape_rule("simplernn")
def _simplernn(p: SimpleRNNParams, inputs: list[Shape]) -> LayerResult:
    return _recurrent(p, inputs, gates=1, bias_rows=1)


@shape_rule("embedding")
def _embedding(p: EmbeddingParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    return LayerResult((*shape, p.output_dim), params=p.input_dim * p.output_dim)


@shape_rule("dropout")
def _dropout(p: DropoutParams, inputs: list[Shape]) -> LayerResult:
    return LayerResult(inputs[0])


@shape_rule("batchnorm")
def _batchnorm(p: BatchNormParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    # gamma and beta are trained; the moving mean and variance are not.
    channels = shape[-1]
    return LayerResult(shape, params=4 * channels, non_trainable=2 * channels)


@shape_rule("reshape")
def _reshape(p: ReshapeParams, inputs: list[Shape]) -> LayerResult:
    (shape,) = inputs
    try:
        target = [int(x) for x in p.target_shape.split(",")]
    except ValueError:
        raise ShapeError(
            DiagnosticCode.INVALID_RESHAPE,
            f"target shape {p.target_shape!r} must be comma-separated whole numbers, for example '7,7,64'.",
        ) from None
    if any(d == 0 or d < -1 for d in target) or target.count(-1) > 1:
        raise ShapeError(
            DiagnosticCode.INVALID_RESHAPE,
            f"target shape {p.target_shape!r} must use positive sizes, with at most one -1 to infer a size.",
        )
    total = _prod(shape)
    if -1 in target:
        known = math.prod(d for d in target if d != -1)
        if total % known:
            raise ShapeError(
                DiagnosticCode.INVALID_RESHAPE,
                f"cannot reshape {total} values from {format_shape(shape)} into {p.target_shape!r}.",
                f"The product of the known sizes ({known}) must divide {total}.",
            )
        target[target.index(-1)] = total // known
    elif math.prod(target) != total:
        raise ShapeError(
            DiagnosticCode.INVALID_RESHAPE,
            f"cannot reshape {format_shape(shape)} ({total} values) into {p.target_shape!r} "
            f"({math.prod(target)} values).",
            f"Choose a target shape whose sizes multiply to {total}.",
        )
    return LayerResult(tuple(target))


@shape_rule("concatenate")
def _concatenate(p: ConcatenateParams, inputs: list[Shape]) -> LayerResult:
    return LayerResult(concat_shapes(inputs, p.axis))
