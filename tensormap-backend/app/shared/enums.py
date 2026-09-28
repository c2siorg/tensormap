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
