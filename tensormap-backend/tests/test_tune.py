"""Unit tests for hyperparameter tuning service."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())


class TestHyperparameterTuning:
    def test_import(self):
        from app.services.deep_learning import tune_hyperparameters_service

        assert callable(tune_hyperparameters_service)

    def test_router_import(self):
        from app.routers.deep_learning import tune_hyperparameters

        assert callable(tune_hyperparameters)
