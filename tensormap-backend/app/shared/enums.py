from enum import IntEnum


class ProblemType(IntEnum):
    """Supported ML problem types for model training."""

    CLASSIFICATION = 1
    REGRESSION = 2
    IMAGE_CLASSIFICATION = 3
