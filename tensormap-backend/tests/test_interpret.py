"""Unit tests for model interpretability service."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())
sys.modules.setdefault("pandas", MagicMock())


class TestInterpretability:
    def test_import(self):
        from app.services.deep_learning import interpret_model_service

        assert callable(interpret_model_service)

    def test_router_import(self):
        from app.routers.deep_learning import interpret_model

        assert callable(interpret_model)
