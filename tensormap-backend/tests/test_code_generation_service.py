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
    _file_location,
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


def _make_model(model_type, file_id=1, target_field="label", training_split=0.8, loss=None):
    model = MagicMock()
    model.file_id = file_id
    model.model_type = model_type
    model.target_field = target_field
    model.training_split = training_split
    model.optimizer = "adam"
    model.metric = "accuracy"
    model.epochs = 10
    model.loss = loss
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

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_configured_loss_is_passed_to_template(self, mock_loader, mock_env, mock_db):
        from app.shared.constants import MODEL_LOSS, MODEL_LOSS_EXPRESSION

        model = _make_model(ProblemType.REGRESSION, loss="huber")
        file = _make_file(file_type="csv", file_name="housing.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_template.render.return_value = "CODE"
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("housing_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data["dl_model"][MODEL_LOSS] == "huber"
        assert rendered_data["dl_model"][MODEL_LOSS_EXPRESSION] == "tf.keras.losses.Huber()"

    def test_default_loss_fallback_when_none_for_regression_and_classification(self, mock_db):
        from app.shared.constants import MODEL_LOSS, MODEL_LOSS_EXPRESSION

        # Regression defaults to MSE
        reg_model = _make_model(ProblemType.REGRESSION, loss=None)
        csv_file = _make_file(file_type="csv", file_name="data.csv")
        mock_db.exec.return_value.first.side_effect = [reg_model, csv_file]

        with (
            patch("app.services.code_generation.Environment") as mock_env,
            patch("app.services.code_generation.FileSystemLoader"),
        ):
            mock_tmpl = MagicMock()
            mock_env.return_value.get_template.return_value = mock_tmpl
            generate_code("reg_model", mock_db)
            data = mock_tmpl.render.call_args.kwargs["data"]
            assert data["dl_model"][MODEL_LOSS] == "mean_squared_error"
            assert data["dl_model"][MODEL_LOSS_EXPRESSION] == "tf.keras.losses.MeanSquaredError()"

        # Classification defaults to Sparse Categorical Crossentropy
        cls_model = _make_model(ProblemType.CLASSIFICATION, loss=None)
        mock_db.exec.return_value.first.side_effect = [cls_model, csv_file]

        with (
            patch("app.services.code_generation.Environment") as mock_env,
            patch("app.services.code_generation.FileSystemLoader"),
        ):
            mock_tmpl = MagicMock()
            mock_env.return_value.get_template.return_value = mock_tmpl
            generate_code("cls_model", mock_db)
            data = mock_tmpl.render.call_args.kwargs["data"]
            assert data["dl_model"][MODEL_LOSS] == "sparse_categorical_crossentropy"
            expected = "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"
            assert data["dl_model"][MODEL_LOSS_EXPRESSION] == expected

    def test_cross_entropy_from_logits_handling_with_output_layer_activations(self, mock_db):
        import tensorflow as tf

        from app.shared.constants import MODEL_LOSS_EXPRESSION

        csv_file = _make_file(file_type="csv", file_name="data.csv")

        # 1. Output layer with softmax -> from_logits=False
        softmax_model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation="softmax")])
        cls_model = _make_model(ProblemType.CLASSIFICATION, loss="sparse_categorical_crossentropy")
        mock_db.exec.return_value.first.side_effect = [cls_model, csv_file]

        with (
            patch("app.services.code_generation.Environment") as mock_env,
            patch("app.services.code_generation.FileSystemLoader"),
        ):
            mock_tmpl = MagicMock()
            mock_env.return_value.get_template.return_value = mock_tmpl
            generate_code("cls_model", mock_db, model=softmax_model)
            data = mock_tmpl.render.call_args.kwargs["data"]
            expected = "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)"
            assert data["dl_model"][MODEL_LOSS_EXPRESSION] == expected

        # 2. Output layer with sigmoid -> from_logits=False
        sigmoid_model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
        bin_model = _make_model(ProblemType.CLASSIFICATION, loss="binary_crossentropy")
        mock_db.exec.return_value.first.side_effect = [bin_model, csv_file]

        with (
            patch("app.services.code_generation.Environment") as mock_env,
            patch("app.services.code_generation.FileSystemLoader"),
        ):
            mock_tmpl = MagicMock()
            mock_env.return_value.get_template.return_value = mock_tmpl
            generate_code("bin_model", mock_db, model=sigmoid_model)
            data = mock_tmpl.render.call_args.kwargs["data"]
            assert data["dl_model"][MODEL_LOSS_EXPRESSION] == "tf.keras.losses.BinaryCrossentropy(from_logits=False)"

        # 3. Output layer linear (logits) -> from_logits=True
        linear_model = tf.keras.Sequential([tf.keras.layers.Dense(5)])
        cat_model = _make_model(ProblemType.CLASSIFICATION, loss="categorical_crossentropy")
        mock_db.exec.return_value.first.side_effect = [cat_model, csv_file]

        with (
            patch("app.services.code_generation.Environment") as mock_env,
            patch("app.services.code_generation.FileSystemLoader"),
        ):
            mock_tmpl = MagicMock()
            mock_env.return_value.get_template.return_value = mock_tmpl
            generate_code("cat_model", mock_db, model=linear_model)
            data = mock_tmpl.render.call_args.kwargs["data"]
            expected = "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"
            assert data["dl_model"][MODEL_LOSS_EXPRESSION] == expected

    def test_loss_to_code_expression_all_losses_and_error_handling(self):
        from app.services.model_run import loss_to_code_expression
        from app.shared.enums import LossFunction

        assert loss_to_code_expression(LossFunction.MEAN_SQUARED_ERROR) == "tf.keras.losses.MeanSquaredError()"
        assert loss_to_code_expression("mean_absolute_error") == "tf.keras.losses.MeanAbsoluteError()"
        assert loss_to_code_expression("huber") == "tf.keras.losses.Huber()"
        assert (
            loss_to_code_expression("binary_crossentropy", from_logits=False)
            == "tf.keras.losses.BinaryCrossentropy(from_logits=False)"
        )
        assert (
            loss_to_code_expression("categorical_crossentropy", from_logits=True)
            == "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"
        )

        with pytest.raises(ValueError, match="Unsupported loss function"):
            loss_to_code_expression("invalid_loss")

    def test_real_templates_render_expected_compile_loss(self, mock_db):
        import tensorflow as tf

        # 1. Linear regression template
        reg_model = _make_model(ProblemType.REGRESSION, loss="huber")
        csv_file = _make_file(file_type="csv", file_name="housing.csv")
        mock_db.exec.return_value.first.side_effect = [reg_model, csv_file]

        code_reg = generate_code("reg_model", mock_db)
        assert "loss=tf.keras.losses.Huber()" in code_reg
        assert 'optimizer="adam"' in code_reg

        # 2. Multi-class classification template with softmax model
        softmax_model = tf.keras.Sequential([tf.keras.layers.Dense(3, activation="softmax")])
        cls_model = _make_model(ProblemType.CLASSIFICATION, loss="categorical_crossentropy")
        mock_db.exec.return_value.first.side_effect = [cls_model, csv_file]

        code_cls = generate_code("cls_model", mock_db, model=softmax_model)
        assert "loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False)" in code_cls

        # 3. Simple image classification template with logits
        img_model = _make_model(ProblemType.IMAGE_CLASSIFICATION, loss="sparse_categorical_crossentropy")
        zip_file = _make_file(file_type="zip", file_name="images.zip")
        image_props = MagicMock(image_size=128, batch_size=32, color_mode="rgb", label_mode="categorical")
        mock_db.exec.return_value.first.side_effect = [img_model, zip_file, image_props]

        code_img = generate_code("img_model", mock_db)
        assert "loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)" in code_img
