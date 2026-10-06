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
