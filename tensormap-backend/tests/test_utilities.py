"""Unit tests for model utilities."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())


class TestModelUtilities:
    def test_import_services(self):
        from app.services.deep_learning import check_model_name_service, get_model_count_service

        assert callable(check_model_name_service)
        assert callable(get_model_count_service)

    def test_import_interpret(self):
        from app.services.deep_learning import interpret_model_service

        assert callable(interpret_model_service)
