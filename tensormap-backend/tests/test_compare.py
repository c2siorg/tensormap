"""Unit tests for run comparison service."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())


class TestCompareRuns:
    def test_import(self):
        from app.services.deep_learning import compare_runs_service

        assert callable(compare_runs_service)

    def test_router_import(self):
        from app.routers.deep_learning import compare_runs

        assert callable(compare_runs)
