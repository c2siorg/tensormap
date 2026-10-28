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

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from app.services.code_generation import (  # noqa: E402
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
    MODEL_EPOCHS,
    MODEL_JSON_FILE,
    MODEL_LOSS,
    MODEL_LOSS_EXPRESSION,
    MODEL_METRIC,
    MODEL_OPTIMIZER,
    MODEL_TRAINING_SPLIT,
    TEMPLATE_ROOT,
)
from app.shared.enums import LossFunction, ProblemType  # noqa: E402

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
# _loss_expression
# ---------------------------------------------------------------------------


class TestLossExpression:
    def test_categorical_crossentropy_expression(self):
        expr = _loss_expression(LossFunction.CATEGORICAL_CROSSENTROPY)
        assert expr == "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"

    def test_binary_crossentropy_expression(self):
        expr = _loss_expression(LossFunction.BINARY_CROSSENTROPY)
        assert expr == "tf.keras.losses.BinaryCrossentropy(from_logits=True)"

    def test_sparse_categorical_crossentropy_expression(self):
        expr = _loss_expression(LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY)
        assert expr == "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"

    def test_cross_entropy_with_from_logits_false(self):
        assert _loss_expression(LossFunction.CATEGORICAL_CROSSENTROPY, from_logits=False) == (
            "tf.keras.losses.CategoricalCrossentropy(from_logits=False)"
        )
        assert _loss_expression(LossFunction.BINARY_CROSSENTROPY, from_logits=False) == (
            "tf.keras.losses.BinaryCrossentropy(from_logits=False)"
        )
        assert _loss_expression(LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY, from_logits=False) == (
            "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)"
        )

    @pytest.mark.parametrize(
        "loss_name",
        [
            LossFunction.MEAN_SQUARED_ERROR,
            LossFunction.MEAN_ABSOLUTE_ERROR,
            LossFunction.HUBER,
            "cosine_similarity",
        ],
    )
    def test_string_losses_quoted(self, loss_name):
        expr = _loss_expression(loss_name)
        assert expr == f'"{loss_name}"'


# ---------------------------------------------------------------------------
# generate_code — loss function context
# ---------------------------------------------------------------------------


class TestGenerateCodeLossContext:
    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_passes_configured_loss(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.CLASSIFICATION, loss=LossFunction.CATEGORICAL_CROSSENTROPY)
        file = _make_file(file_type="csv", file_name="data.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("cat_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "categorical_crossentropy"
        assert (
            rendered_data[LEARNING_MODEL][MODEL_LOSS_EXPRESSION]
            == "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"
        )

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_with_softmax_activation_sets_from_logits_false(self, mock_loader, mock_env, mock_db):
        softmax_graph = {"config": {"layers": [{"class_name": "Dense", "config": {"activation": "softmax"}}]}}
        model = _make_model(
            ProblemType.CLASSIFICATION,
            loss=LossFunction.CATEGORICAL_CROSSENTROPY,
            graph_json=softmax_graph,
        )
        file = _make_file(file_type="csv", file_name="data.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("softmax_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert (
            rendered_data[LEARNING_MODEL][MODEL_LOSS_EXPRESSION]
            == "tf.keras.losses.CategoricalCrossentropy(from_logits=False)"
        )

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_with_sigmoid_activation_sets_from_logits_false(self, mock_loader, mock_env, mock_db):
        sigmoid_graph = {"config": {"layers": [{"class_name": "Dense", "config": {"activation": "sigmoid"}}]}}
        model = _make_model(
            ProblemType.CLASSIFICATION,
            loss=LossFunction.BINARY_CROSSENTROPY,
            graph_json=sigmoid_graph,
        )
        file = _make_file(file_type="csv", file_name="data.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("sigmoid_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert (
            rendered_data[LEARNING_MODEL][MODEL_LOSS_EXPRESSION]
            == "tf.keras.losses.BinaryCrossentropy(from_logits=False)"
        )

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_classification_defaults_to_sparse_categorical_when_none(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.CLASSIFICATION, loss=None)
        file = _make_file(file_type="csv", file_name="data.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("default_cls_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "sparse_categorical_crossentropy"
        assert (
            rendered_data[LEARNING_MODEL][MODEL_LOSS_EXPRESSION]
            == "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"
        )

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_regression_passes_configured_loss(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.REGRESSION, loss=LossFunction.HUBER)
        file = _make_file(file_type="csv", file_name="prices.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("huber_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "huber"

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_regression_defaults_to_mse_when_none(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.REGRESSION, loss=None)
        file = _make_file(file_type="csv", file_name="prices.csv")
        mock_db.exec.return_value.first.side_effect = [model, file]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("default_reg_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "mean_squared_error"

    @patch("app.services.code_generation.Environment")
    @patch("app.services.code_generation.FileSystemLoader")
    def test_image_classification_passes_configured_loss(self, mock_loader, mock_env, mock_db):
        model = _make_model(ProblemType.IMAGE_CLASSIFICATION, loss=LossFunction.BINARY_CROSSENTROPY)
        file = _make_file(file_type="zip", file_name="images.zip")
        mock_db.exec.return_value.first.side_effect = [model, file, None]

        mock_template = MagicMock()
        mock_env.return_value.get_template.return_value = mock_template

        generate_code("binary_img_model", mock_db)

        rendered_data = mock_template.render.call_args.kwargs["data"]
        assert rendered_data[LEARNING_MODEL][MODEL_LOSS] == "binary_crossentropy"
        assert (
            rendered_data[LEARNING_MODEL][MODEL_LOSS_EXPRESSION]
            == "tf.keras.losses.BinaryCrossentropy(from_logits=True)"
        )


# ---------------------------------------------------------------------------
# Template Rendering Parity Tests
# ---------------------------------------------------------------------------


class TestTemplateRenderLoss:
    @pytest.fixture()
    def template_env(self):
        return Environment(loader=FileSystemLoader(searchpath=TEMPLATE_ROOT))

    @pytest.mark.parametrize(
        ("loss_name", "expected_snippet"),
        [
            ("categorical_crossentropy", "loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)"),
            ("binary_crossentropy", "loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)"),
            ("sparse_categorical_crossentropy", "loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"),
        ],
    )
    def test_classification_template_renders_correct_loss(self, template_env, loss_name, expected_snippet):
        template = template_env.get_template(CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py")
        data = {
            DATASET: {FILE_NAME: "data.csv", FILE_TARGET: "label", MODEL_TRAINING_SPLIT: 80},
            LEARNING_MODEL: {
                MODEL_JSON_FILE: "model.json",
                MODEL_OPTIMIZER: "adam",
                MODEL_METRIC: "accuracy",
                MODEL_EPOCHS: 5,
                MODEL_LOSS: loss_name,
                MODEL_LOSS_EXPRESSION: _loss_expression(loss_name),
            },
        }
        rendered = template.render(data=data)
        assert expected_snippet in rendered

    @pytest.mark.parametrize(
        ("loss_name", "expected_snippet"),
        [
            ("mean_squared_error", 'loss="mean_squared_error"'),
            ("mean_absolute_error", 'loss="mean_absolute_error"'),
            ("huber", 'loss="huber"'),
        ],
    )
    def test_regression_template_renders_correct_loss(self, template_env, loss_name, expected_snippet):
        template = template_env.get_template(CODE_TEMPLATE_FOLDER + "linear-regression-all-float.py")
        data = {
            DATASET: {FILE_NAME: "data.csv", FILE_TARGET: "target", MODEL_TRAINING_SPLIT: 80},
            LEARNING_MODEL: {
                MODEL_JSON_FILE: "model.json",
                MODEL_OPTIMIZER: "adam",
                MODEL_METRIC: "mae",
                MODEL_EPOCHS: 5,
                MODEL_LOSS: loss_name,
                MODEL_LOSS_EXPRESSION: _loss_expression(loss_name),
            },
        }
        rendered = template.render(data=data)
        assert expected_snippet in rendered

    def test_image_classification_template_renders_correct_loss(self, template_env):
        template = template_env.get_template(CODE_TEMPLATE_FOLDER + "simple-image-classification.py")
        data = {
            DATASET: {
                FILE_NAME: "images",
                MODEL_TRAINING_SPLIT: 80,
                IMG_SIZE: 128,
                BATCH_SIZE: 32,
                COLOR_MODE: "rgb",
                LABEL_MODE: "categorical",
            },
            LEARNING_MODEL: {
                MODEL_JSON_FILE: "model.json",
                MODEL_OPTIMIZER: "adam",
                MODEL_METRIC: "accuracy",
                MODEL_EPOCHS: 10,
                MODEL_LOSS: "categorical_crossentropy",
                MODEL_LOSS_EXPRESSION: _loss_expression("categorical_crossentropy"),
            },
        }
        rendered = template.render(data=data)
        assert "loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)" in rendered


# ---------------------------------------------------------------------------
# _emits_logits and _model_emits_logits edge cases
# ---------------------------------------------------------------------------


class TestModelEmitsLogitsEdgeCases:
    def test_activation_layer_relu_emits_logits(self):
        """A final Activation layer with relu emits raw logits, not probabilities."""
        graph = {"config": {"layers": [{"class_name": "Activation", "config": {"activation": "relu"}}]}}
        model = _make_model(ProblemType.CLASSIFICATION, graph_json=graph)
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("relu_act_model", model) is True

    def test_activation_layer_tanh_emits_logits(self):
        """A final Activation layer with tanh emits raw values, not probabilities."""
        graph = {"config": {"layers": [{"class_name": "Activation", "config": {"activation": "tanh"}}]}}
        model = _make_model(ProblemType.CLASSIFICATION, graph_json=graph)
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("tanh_act_model", model) is True

    def test_activation_layer_softmax_emits_probabilities(self):
        """A final Activation layer with softmax emits probabilities (from_logits=False)."""
        graph = {"config": {"layers": [{"class_name": "Activation", "config": {"activation": "softmax"}}]}}
        model = _make_model(ProblemType.CLASSIFICATION, graph_json=graph)
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("softmax_act_model", model) is False

    def test_softmax_class_layer_emits_probabilities(self):
        """A final Softmax layer emits probabilities (from_logits=False)."""
        graph = {"config": {"layers": [{"class_name": "Softmax", "config": {}}]}}
        model = _make_model(ProblemType.CLASSIFICATION, graph_json=graph)
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("softmax_class_model", model) is False

    def test_dense_linear_or_relu_emits_logits(self):
        """A final Dense layer with linear or relu activation emits logits."""
        graph_linear = {"config": {"layers": [{"class_name": "Dense", "config": {"activation": "linear"}}]}}
        model_linear = _make_model(ProblemType.CLASSIFICATION, graph_json=graph_linear)
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("linear_model", model_linear) is True

        graph_relu = {"config": {"layers": [{"class_name": "Dense", "config": {"activation": "relu"}}]}}
        model_relu = _make_model(ProblemType.CLASSIFICATION, graph_json=graph_relu)
        assert _model_emits_logits("relu_model", model_relu) is True

    def test_functional_model_resolves_output_layer_by_name(self):
        """Functional models specify output_layers; resolution must find that layer."""
        graph = {
            "config": {
                "layers": [
                    {"class_name": "Dense", "config": {"name": "final_out", "activation": "softmax"}},
                    {"class_name": "Dense", "config": {"name": "aux_dense", "activation": "relu"}},
                ],
                "output_layers": [["final_out", 0, 0]],
            }
        }
        model = _make_model(ProblemType.CLASSIFICATION, graph_json=graph)
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("functional_model", model) is False

    def test_malformed_graph_json_logs_warning_and_defaults_to_true(self, caplog):
        """Malformed JSON string logs a warning and falls back to from_logits=True."""
        model = _make_model(ProblemType.CLASSIFICATION, graph_json="{invalid json syntax")
        from app.services.code_generation import _model_emits_logits

        with caplog.at_level("WARNING"):
            result = _model_emits_logits("bad_json_model", model)
        assert result is True
        assert any("Failed to parse model JSON" in r.message for r in caplog.records)

    def test_empty_graph_json_defaults_to_true(self):
        """Empty or layerless model config defaults to from_logits=True."""
        model = _make_model(ProblemType.CLASSIFICATION, graph_json={})
        from app.services.code_generation import _model_emits_logits

        assert _model_emits_logits("empty_model", model) is True

    def test_reads_model_json_from_disk_file(self, tmp_path, monkeypatch):
        """When the JSON file exists on disk, it is read and evaluated."""
        import json

        from app.services import code_generation

        model_file = tmp_path / "disk_model.json"
        model_file.write_text(
            json.dumps({"config": {"layers": [{"class_name": "Dense", "config": {"activation": "softmax"}}]}})
        )
        monkeypatch.setattr(code_generation, "MODEL_GENERATION_LOCATION", str(tmp_path) + "/")

        model = _make_model(ProblemType.CLASSIFICATION, graph_json=None)
        assert code_generation._model_emits_logits("disk_model", model) is False

    def test_disk_file_oserror_logs_warning_and_falls_back(self, monkeypatch, caplog):
        import os

        """An OSError reading the disk file logs a warning and falls back to graph_json."""
        from app.services import code_generation

        monkeypatch.setattr(os.path, "exists", lambda p: True)

        def mock_open(*args, **kwargs):
            raise OSError("Permission denied")

        model = _make_model(
            ProblemType.CLASSIFICATION,
            graph_json={"config": {"layers": [{"class_name": "Dense", "config": {"activation": "sigmoid"}}]}},
        )

        with monkeypatch.context() as m:
            m.setattr("builtins.open", mock_open)
            with caplog.at_level("WARNING"):
                result = code_generation._model_emits_logits("oserror_model", model)

        assert result is False
        assert any("Failed to read model JSON file" in r.message for r in caplog.records)


class TestEmitsLogitsSharedHelper:
    def test_emits_logits_with_keras_model_mock_softmax(self):
        """_emits_logits works with genuine/mock Model instances ending in softmax."""
        from app.services.model_run import _emits_logits

        layer = MagicMock()
        act = MagicMock()
        act.__name__ = "softmax"
        layer.activation = act

        model = MagicMock()
        model.layers = [layer]
        assert _emits_logits(model) is False

    def test_emits_logits_with_keras_model_mock_relu(self):
        """_emits_logits works with Model instances ending in relu."""
        from app.services.model_run import _emits_logits

        layer = MagicMock()
        act = MagicMock()
        act.__name__ = "relu"
        layer.activation = act

        model = MagicMock()
        model.layers = [layer]
        assert _emits_logits(model) is True

    def test_emits_logits_with_keras_model_mock_softmax_class(self):
        """_emits_logits works with Model instances ending in a Softmax layer."""
        from app.services.model_run import _emits_logits

        class Softmax:
            pass

        layer = Softmax()
        model = MagicMock()
        model.layers = [layer]
        assert _emits_logits(model) is False


class TestTemplateOneHotEncoding:
    @pytest.fixture()
    def template_env(self):
        return Environment(loader=FileSystemLoader(searchpath=TEMPLATE_ROOT))

    def test_multi_class_template_one_hot_encodes_for_categorical_crossentropy(self, template_env):
        template = template_env.get_template(CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py")
        data = {
            DATASET: {FILE_NAME: "data.csv", FILE_TARGET: "label", MODEL_TRAINING_SPLIT: 80},
            LEARNING_MODEL: {
                MODEL_JSON_FILE: "model.json",
                MODEL_OPTIMIZER: "adam",
                MODEL_METRIC: "accuracy",
                MODEL_EPOCHS: 5,
                MODEL_LOSS: "categorical_crossentropy",
                MODEL_LOSS_EXPRESSION: "tf.keras.losses.CategoricalCrossentropy(from_logits=True)",
            },
        }
        rendered = template.render(data=data)
        assert 'if "categorical_crossentropy" == "categorical_crossentropy":' in rendered
        assert "y = pd.get_dummies(y)" in rendered

    def test_multi_class_template_raw_labels_for_sparse_categorical_crossentropy(self, template_env):
        template = template_env.get_template(CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py")
        data = {
            DATASET: {FILE_NAME: "data.csv", FILE_TARGET: "label", MODEL_TRAINING_SPLIT: 80},
            LEARNING_MODEL: {
                MODEL_JSON_FILE: "model.json",
                MODEL_OPTIMIZER: "adam",
                MODEL_METRIC: "accuracy",
                MODEL_EPOCHS: 5,
                MODEL_LOSS: "sparse_categorical_crossentropy",
                MODEL_LOSS_EXPRESSION: "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)",
            },
        }
        rendered = template.render(data=data)
        assert 'y = features["label"]' in rendered
        assert 'if "sparse_categorical_crossentropy" == "categorical_crossentropy":' in rendered
