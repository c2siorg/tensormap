"""Unit tests for generate_code() happy paths and _map_template() in code_generation.py.

Covers the paths test_null_checks.py does not: successful code generation for
CLASSIFICATION, REGRESSION, and IMAGE_CLASSIFICATION problem types, the
_map_template() enum-to-path mapping, and confirms _file_location()/dataset
context data is assembled correctly before being handed to Jinja2.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy third-party modules before importing the services
_tf_stub = MagicMock()
sys.modules.setdefault("tensorflow", _tf_stub)
sys.modules.setdefault("flatten_json", MagicMock())

from app.services.code_generation import (  # noqa: E402
    _map_template,
    generate_code,
)
from app.shared.constants import (  # noqa: E402
    BATCH_SIZE,
    CODE_TEMPLATE_FOLDER,
    COLOR_MODE,
    DATASET,
    FILE_NAME,
    FILE_TARGET,
    IMG_SIZE,
    LABEL_MODE,
    MODEL_TRAINING_SPLIT,
)
from app.shared.enums import ProblemType  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.exec.return_value.first.return_value = None
    return db


def _make_model(model_type, file_id=1, target_field="label", training_split=0.8):
    model = MagicMock()
    model.file_id = file_id
    model.model_type = model_type
    model.target_field = target_field
    model.training_split = training_split
    model.optimizer = "adam"
    model.metric = "accuracy"
    model.epochs = 10
    return model


def _make_file(file_type="csv", file_name="iris.csv"):
    f = MagicMock()
    f.file_type = file_type
    f.file_name = file_name
    return f


# ---------------------------------------------------------------------------
# _map_template
# ---------------------------------------------------------------------------


class TestMapTemplate:
    def test_classification_maps_to_correct_template(self):
        result = _map_template(ProblemType.CLASSIFICATION)
        assert result == CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py"

    def test_regression_maps_to_correct_template(self):
        result = _map_template(ProblemType.REGRESSION)
        assert result == CODE_TEMPLATE_FOLDER + "linear-regression-all-float.py"

    def test_image_classification_maps_to_correct_template(self):
        result = _map_template(ProblemType.IMAGE_CLASSIFICATION)
        assert result == CODE_TEMPLATE_FOLDER + "simple-image-classification.py"

    def test_none_model_type_raises_value_error_naming_it(self):
        with pytest.raises(ValueError, match="Unknown model type: None"):
            _map_template(None)

    def test_unrecognised_model_type_raises_value_error_naming_it(self):
        with pytest.raises(ValueError, match="Unknown model type: 99"):
            _map_template(99)


# ---------------------------------------------------------------------------
# generate_code — happy paths
# ---------------------------------------------------------------------------


class TestGenerateCodeHappyPaths:
    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_renders_with_correct_dataset_context(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.CLASSIFICATION)
        file = _make_file(file_type="csv", file_name="iris.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_template.render.return_value = "GENERATED_CODE"
        mock_env.return_value.get_template.return_value = mock_template

        result = generate_code("iris_model", mock_db)

        assert result == "GENERATED_CODE"
        mock_env.return_value.get_template.assert_called_once_with(
            CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py"
        )
        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[DATASET][FILE_NAME] == "iris.csv"
        assert rendered_data[DATASET][FILE_TARGET] == "label"
        assert rendered_data[DATASET][MODEL_TRAINING_SPLIT] == 0.8

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_regression_renders_with_correct_template(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.REGRESSION, target_field="price", training_split=0.75)
        file = _make_file(file_type="csv", file_name="housing.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_template.render.return_value = "REGRESSION_CODE"
        mock_env.return_value.get_template.return_value = mock_template

        result = generate_code("housing_model", mock_db)

        assert result == "REGRESSION_CODE"
        mock_env.return_value.get_template.assert_called_once_with(
            CODE_TEMPLATE_FOLDER + "linear-regression-all-float.py"
        )
        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[DATASET][FILE_TARGET] == "price"
        assert rendered_data[DATASET][MODEL_TRAINING_SPLIT] == 0.75

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_image_classification_includes_image_properties(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.IMAGE_CLASSIFICATION, target_field=None)
        file = _make_file(file_type="zip", file_name="images.zip")

        image_props = MagicMock()
        image_props.image_size = 128
        image_props.batch_size = 32
        image_props.color_mode = "rgb"
        image_props.label_mode = "categorical"

        # Order of db.exec().first() calls: model, file, image_properties
        mock_db.exec.return_value.first.side_effect = [model, file, image_props]

        mock_template = MagicMock()
        mock_template.render.return_value = "IMAGE_CODE"
        mock_env.return_value.get_template.return_value = mock_template

        result = generate_code("image_model", mock_db)

        assert result == "IMAGE_CODE"
        mock_env.return_value.get_template.assert_called_once_with(
            CODE_TEMPLATE_FOLDER + "simple-image-classification.py"
        )
        rendered_data = mock_template.render.call_args.kwargs["data"]
        # Zip file -> _file_location strips extension
        assert rendered_data[DATASET][FILE_NAME] == "images"
        assert rendered_data[DATASET][IMG_SIZE] == 128
        assert rendered_data[DATASET][BATCH_SIZE] == 32
        assert rendered_data[DATASET][COLOR_MODE] == "rgb"
        assert rendered_data[DATASET][LABEL_MODE] == "categorical"

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_zip_file_without_image_properties_skips_image_fields(self, mock_loader, mock_env, mock_db):
        """Zip file with no matching ImageProperties row should not crash and
        should not add image-specific keys to the dataset context."""
        model = _make_model(ProblemType.IMAGE_CLASSIFICATION)
        file = _make_file(file_type="zip", file_name="images.zip")

        # image_properties lookup returns None
        mock_db.exec.return_value.first.side_effect = [model, file, None]

        mock_template = MagicMock()
        mock_template.render.return_value = "CODE"
        mock_env.return_value.get_template.return_value = mock_template

        result = generate_code("image_model_no_props", mock_db)

        assert result == "CODE"
        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert IMG_SIZE not in rendered_data[DATASET]
        assert BATCH_SIZE not in rendered_data[DATASET]
        assert COLOR_MODE not in rendered_data[DATASET]
        assert LABEL_MODE not in rendered_data[DATASET]
