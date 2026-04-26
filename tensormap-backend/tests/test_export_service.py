"""Unit tests for model export service."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())


class TestExportModelService:
    def test_import(self):
        from app.services.deep_learning import export_model_service

        assert callable(export_model_service)

    def test_router_import(self):
        from app.routers.deep_learning import export_model

        assert callable(export_model)
