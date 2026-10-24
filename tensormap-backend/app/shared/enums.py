from enum import IntEnum, StrEnum


class ProblemType(IntEnum):
    """Supported ML problem types for model training."""

    CLASSIFICATION = 1
    REGRESSION = 2
    IMAGE_CLASSIFICATION = 3


class LossFunction(StrEnum):
    """Keras loss functions a user can pick on the Training page."""

    SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy"
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    BINARY_CROSSENTROPY = "binary_crossentropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    HUBER = "huber"


CLASSIFICATION_LOSSES = frozenset(
    {
        LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY,
        LossFunction.CATEGORICAL_CROSSENTROPY,
        LossFunction.BINARY_CROSSENTROPY,
    }
)
REGRESSION_LOSSES = frozenset({LossFunction.MEAN_SQUARED_ERROR, LossFunction.MEAN_ABSOLUTE_ERROR, LossFunction.HUBER})


def losses_for_problem_type(problem_type: int) -> frozenset[LossFunction]:
    """Losses that make sense for a problem type; the Training page offers all of them."""
    if problem_type in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
        return CLASSIFICATION_LOSSES
    return REGRESSION_LOSSES


def default_loss_for_problem_type(problem_type: int) -> LossFunction:
    """Default loss function for a given problem type."""
    if problem_type in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
        return LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY
    return LossFunction.MEAN_SQUARED_ERROR


def loss_to_code_expression(loss: str | LossFunction, from_logits: bool = True) -> str:
    """Return the Python expression string for compiling a model with the specified loss."""
    loss_val = LossFunction(loss).value
    logits_str = str(from_logits)
    loss_expressions = {
        LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY.value: (
            f"tf.keras.losses.SparseCategoricalCrossentropy(from_logits={logits_str})"
        ),
        LossFunction.CATEGORICAL_CROSSENTROPY.value: (
            f"tf.keras.losses.CategoricalCrossentropy(from_logits={logits_str})"
        ),
        LossFunction.BINARY_CROSSENTROPY.value: (f"tf.keras.losses.BinaryCrossentropy(from_logits={logits_str})"),
        LossFunction.MEAN_SQUARED_ERROR.value: '"mean_squared_error"',
        LossFunction.MEAN_ABSOLUTE_ERROR.value: '"mean_absolute_error"',
        LossFunction.HUBER.value: "tf.keras.losses.Huber()",
    }
    return loss_expressions.get(loss_val, '"mean_squared_error"')
