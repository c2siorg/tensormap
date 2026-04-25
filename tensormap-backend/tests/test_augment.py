"""Unit tests for tabular augmentation service."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())
sys.modules.setdefault("pandas", MagicMock())


class TestTabularAugmentation:
    def test_import(self):
        from app.services.data_process import augment_tabular_data_service

        assert callable(augment_tabular_data_service)

    def test_router_import(self):
        from app.routers.data_process import augment

        assert callable(augment)
