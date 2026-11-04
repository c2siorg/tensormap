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
    _determine_from_logits,
    _file_location,
    _loss_expression,
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
    LEARNING_MODEL,
    MODEL_LOSS,
    MODEL_TRAINING_SPLIT,
)
from app.shared.enums import LossFunction, ProblemType, default_loss_for_problem_type  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.exec.return_value.first.return_value = None
    return db


def _make_model(model_type, file_id=1, target_field="label", training_split=0.8, loss=None, graph_json=None):
    model = MagicMock()
    model.file_id = file_id
    model.model_type = model_type
    model.target_field = target_field
    model.training_split = training_split
    model.optimizer = "adam"
    model.metric = "accuracy"
    model.epochs = 10
    model.loss = loss
    model.graph_json = graph_json
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
# _file_location
# ---------------------------------------------------------------------------


class TestFileLocation:
    def test_csv_file_without_extension_appends_csv(self):
        file = _make_file(file_type="csv", file_name="iris")
        assert _file_location(file) == "iris.csv"

    def test_csv_file_with_extension_preserves_csv(self):
        file = _make_file(file_type="csv", file_name="iris.csv")
        assert _file_location(file) == "iris.csv"

    def test_zip_file_strips_extension(self):
        file = _make_file(file_type="zip", file_name="images.zip")
        assert _file_location(file) == "images"


# ---------------------------------------------------------------------------
# generate_code — happy paths
# ---------------------------------------------------------------------------


class TestGenerateCodeHappyPaths:
    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_appends_csv_extension_when_file_name_lacks_it(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.CLASSIFICATION)
        # Database stores CSV file_name without extension
        file = _make_file(file_type="csv", file_name="iris")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_template.render.return_value = "GENERATED_CODE"
        mock_env.return_value.get_template.return_value = mock_template

        result = generate_code("iris_model", mock_db)

        assert result == "GENERATED_CODE"
        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[DATASET][FILE_NAME] == "iris.csv"

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


# ---------------------------------------------------------------------------
# Loss resolution and rendering
# ---------------------------------------------------------------------------


class TestLossExpression:
    def test_default_loss_classification(self):
        expr = _loss_expression(None, ProblemType.CLASSIFICATION)
        assert expr == "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"

    def test_default_loss_image_classification(self):
        expr = _loss_expression(None, ProblemType.IMAGE_CLASSIFICATION)
        assert expr == "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"

    def test_default_loss_regression(self):
        expr = _loss_expression(None, ProblemType.REGRESSION)
        assert expr == "tf.keras.losses.MeanSquaredError()"

    def test_categorical_crossentropy_from_logits_true(self):
        expr = _loss_expression(LossFunction.CATEGORICAL_CROSSENTROPY, ProblemType.CLASSIFICATION, from_logits=True)
        assert expr == "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"

    def test_categorical_crossentropy_from_logits_false(self):
        expr = _loss_expression(LossFunction.CATEGORICAL_CROSSENTROPY, ProblemType.CLASSIFICATION, from_logits=False)
        assert expr == "tf.keras.losses.CategoricalCrossentropy(from_logits=False)"

    def test_binary_crossentropy(self):
        expr = _loss_expression(LossFunction.BINARY_CROSSENTROPY, ProblemType.CLASSIFICATION)
        assert expr == "tf.keras.losses.BinaryCrossentropy(from_logits=True)"

    def test_mean_absolute_error(self):
        expr = _loss_expression(LossFunction.MEAN_ABSOLUTE_ERROR, ProblemType.REGRESSION)
        assert expr == "tf.keras.losses.MeanAbsoluteError()"

    def test_huber(self):
        expr = _loss_expression(LossFunction.HUBER, ProblemType.REGRESSION)
        assert expr == "tf.keras.losses.Huber()"

    def test_invalid_loss_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown loss function: 'invalid_loss'"):
            _loss_expression("invalid_loss", ProblemType.CLASSIFICATION)

    def test_determine_from_logits_softmax(self):
        model = MagicMock()
        model.graph_json = {
            "config": {
                "layers": [
                    {"config": {"name": "dense_1", "activation": "relu"}},
                    {"config": {"name": "dense_2", "activation": "softmax"}},
                ]
            }
        }
        assert _determine_from_logits(model) is False

    def test_determine_from_logits_linear(self):
        model = MagicMock()
        model.graph_json = {
            "config": {
                "layers": [
                    {"config": {"name": "dense_1", "activation": "relu"}},
                    {"config": {"name": "dense_2", "activation": "linear"}},
                ]
            }
        }
        assert _determine_from_logits(model) is True

    def test_default_loss_for_problem_type_helper(self):
        assert default_loss_for_problem_type(ProblemType.CLASSIFICATION) == LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY
        assert (
            default_loss_for_problem_type(ProblemType.IMAGE_CLASSIFICATION)
            == LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY
        )
        assert default_loss_for_problem_type(ProblemType.REGRESSION) == LossFunction.MEAN_SQUARED_ERROR


class TestGenerateCodeLossRendering:
    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_passes_loss_to_template(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.CLASSIFICATION, loss=LossFunction.CATEGORICAL_CROSSENTROPY)
        file = _make_file(file_type="csv", file_name="iris.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_template.render.return_value = "CODE"
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("cat_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_regression_passes_huber_loss_to_template(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.REGRESSION, loss=LossFunction.HUBER)
        file = _make_file(file_type="csv", file_name="prices.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_template.render.return_value = "CODE"
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("huber_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "tf.keras.losses.Huber()"

    def test_actual_regression_template_renders_huber_loss(self, mock_db):
        model = _make_model(ProblemType.REGRESSION, loss=LossFunction.HUBER)
        file = _make_file(file_type="csv", file_name="prices.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        rendered_code = generate_code("real_huber_model", mock_db)
        assert "loss=tf.keras.losses.Huber()," in rendered_code
        # Ensure it compiles without SyntaxError
        compile(rendered_code, "<string>", "exec")

    def test_actual_classification_template_renders_categorical_crossentropy(self, mock_db):
        model = _make_model(ProblemType.CLASSIFICATION, loss=LossFunction.CATEGORICAL_CROSSENTROPY)
        file = _make_file(file_type="csv", file_name="iris.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        rendered_code = generate_code("real_cat_model", mock_db)
        assert "loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)," in rendered_code
        compile(rendered_code, "<string>", "exec")

    def test_actual_image_template_renders_binary_crossentropy(self, mock_db):
        model = _make_model(ProblemType.IMAGE_CLASSIFICATION, loss=LossFunction.BINARY_CROSSENTROPY)
        file = _make_file(file_type="zip", file_name="cats_dogs.zip")
        image_props = MagicMock(image_size=128, batch_size=32, color_mode="rgb", label_mode="binary")
        mock_db.exec.return_value.first.side_effect = [model, file, image_props]

        rendered_code = generate_code("real_image_model", mock_db)
        assert "loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)," in rendered_code
        compile(rendered_code, "<string>", "exec")
