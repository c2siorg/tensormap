"""Canonical layer registry exposed to API clients.

This module centralizes layer metadata so frontend forms and backend
validation can rely on a shared contract.
"""

from typing import Any

_LAYER_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "custominput",
        "label": "Input",
        "category": "core",
        "compiler_supported": True,
        "params": [
            {"name": "dim-1", "type": "int", "required": True, "default": 28, "min": 1},
            {"name": "dim-2", "type": "int", "required": False, "default": 28, "min": 0},
            {"name": "dim-3", "type": "int", "required": False, "default": 1, "min": 0},
        ],
    },
    {
        "type": "customdense",
        "label": "Dense",
        "category": "core",
        "compiler_supported": True,
        "params": [
            {"name": "units", "type": "int", "required": True, "default": 64, "min": 1},
            {
                "name": "activation",
                "type": "select",
                "required": True,
                "default": "relu",
                "options": ["none", "relu", "sigmoid", "tanh", "softmax", "elu", "selu"],
            },
        ],
    },
    {
        "type": "customflatten",
        "label": "Flatten",
        "category": "core",
        "compiler_supported": True,
        "params": [],
    },
    {
        "type": "customconv",
        "label": "Conv2D",
        "category": "cnn",
        "compiler_supported": True,
        "params": [
            {"name": "filter", "type": "int", "required": True, "default": 16, "min": 1},
            {"name": "kernelX", "type": "int", "required": True, "default": 3, "min": 1},
            {"name": "kernelY", "type": "int", "required": True, "default": 3, "min": 1},
            {"name": "strideX", "type": "int", "required": True, "default": 1, "min": 1},
            {"name": "strideY", "type": "int", "required": True, "default": 1, "min": 1},
            {
                "name": "padding",
                "type": "select",
                "required": True,
                "default": "valid",
                "options": ["valid", "same"],
            },
            {
                "name": "activation",
                "type": "select",
                "required": True,
                "default": "relu",
                "options": ["none", "relu", "sigmoid", "tanh", "softmax", "elu", "selu"],
            },
        ],
    },
    {
        "type": "customdropout",
        "label": "Dropout",
        "category": "regularization",
        "compiler_supported": True,
        "params": [
            {"name": "rate", "type": "float", "required": True, "default": 0.2, "min": 0.0, "max": 0.999},
        ],
    },
]


def get_layer_schema() -> list[dict[str, Any]]:
    """Return canonical layer schema metadata for API clients."""
    return _LAYER_SCHEMA
