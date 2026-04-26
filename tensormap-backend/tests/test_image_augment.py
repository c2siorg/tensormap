"""Unit tests for image augmentation service."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())
sys.modules.setdefault("pandas", MagicMock())


class TestImageAugmentation:
    def test_import(self):
        from app.services.data_process import augment_image_service

        assert callable(augment_image_service)

    def test_router_import(self):
        from app.routers.data_process import augment_image

        assert callable(augment_image)
