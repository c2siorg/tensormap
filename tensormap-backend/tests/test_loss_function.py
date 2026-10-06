"""Tests for the user-selectable loss function (Issue #142).

Covers the name-to-Keras mapping in model_run, the request schema validation,
and the problem-type default applied when a training config omits the loss.
"""

from unittest.mock import MagicMock

import pytest
import tensorflow as tf
from pydantic import ValidationError

from app.models.ml import ModelBasic
from app.schemas.deep_learning import TrainingConfigRequest
from app.services.deep_learning import _default_loss, update_training_config_service
from app.services.model_run import _LOSS_CLASSES, _emits_logits, resolve_loss
from app.shared.enums import LossFunction, ProblemType, losses_for_problem_type

EXPECTED_LOSS_CLASSES = {
    "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy,
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
    "binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
    "mean_squared_error": tf.keras.losses.MeanSquaredError,
    "mean_absolute_error": tf.keras.losses.MeanAbsoluteError,
    "huber": tf.keras.losses.Huber,
}


def _training_config(**overrides) -> dict:
    config = {
        "model_name": "my_model",
        "file_id": "0d2a3f1e-7c4b-4b2a-9f2e-1c3d4e5f6a7b",
        "target_field": "label",
        "training_split": 80,
        "problem_type_id": ProblemType.CLASSIFICATION,
        "optimizer": "adam",
        "metric": "accuracy",
        "epochs": 5,
    }
    config.update(overrides)
    return config


def _model_with_output(activation) -> tf.keras.Model:
    return tf.keras.Sequential([tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(3, activation=activation)])


# ---------------------------------------------------------------------------
# resolve_loss / _emits_logits
# ---------------------------------------------------------------------------


class TestResolveLoss:
    def test_mapping_covers_every_supported_name(self):
        assert set(_LOSS_CLASSES) == set(LossFunction) == set(EXPECTED_LOSS_CLASSES)

    @pytest.mark.parametrize(("name", "loss_cls"), EXPECTED_LOSS_CLASSES.items())
    def test_resolves_each_name_to_its_keras_loss(self, name, loss_cls):
        assert isinstance(resolve_loss(name, from_logits=True), loss_cls)

    @pytest.mark.parametrize(
        "name",
        ["sparse_categorical_crossentropy", "categorical_crossentropy", "binary_crossentropy"],
    )
    @pytest.mark.parametrize("from_logits", [True, False])
    def test_crossentropy_losses_take_from_logits(self, name, from_logits):
        assert resolve_loss(name, from_logits=from_logits).from_logits is from_logits

    @pytest.mark.parametrize("name", ["mse", "MeanSquaredError", "", None])
    def test_unknown_name_raises_instead_of_falling_back(self, name):
        with pytest.raises(ValueError, match="Unsupported loss function"):
            resolve_loss(name, from_logits=True)


class TestEmitsLogits:
    @pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
    def test_false_when_output_layer_already_normalises(self, activation):
        assert _emits_logits(_model_with_output(activation)) is False

    @pytest.mark.parametrize("activation", [None, "linear", "relu"])
    def test_true_for_raw_outputs(self, activation):
        assert _emits_logits(_model_with_output(activation)) is True

    def test_false_for_standalone_softmax_layer(self):
        model = tf.keras.Sequential([tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(3), tf.keras.layers.Softmax()])
        assert _emits_logits(model) is False


# ---------------------------------------------------------------------------
# TrainingConfigRequest
# ---------------------------------------------------------------------------


class TestTrainingConfigRequestLoss:
    @pytest.mark.parametrize("name", list(EXPECTED_LOSS_CLASSES))
    def test_accepts_each_supported_name(self, name):
        request = TrainingConfigRequest(**_training_config(loss=name))
        assert request.loss == name

    def test_loss_is_optional(self):
        assert TrainingConfigRequest(**_training_config()).loss is None

    def test_rejects_unknown_name(self):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfigRequest(**_training_config(loss="mse"))
        (error,) = exc_info.value.errors()
        assert error["loc"] == ("loss",)
        assert "sparse_categorical_crossentropy" in error["msg"]


def test_training_config_endpoint_rejects_unknown_loss(client):
    resp = client.patch("/api/v1/model/training-config", json=_training_config(loss="mse"))
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["data"][0]["loc"] == ["body", "loss"]


# ---------------------------------------------------------------------------
# update_training_config_service
# ---------------------------------------------------------------------------


def _db_returning(model: ModelBasic) -> MagicMock:
    db = MagicMock()
    db.exec.return_value.first.return_value = model
    return db


class TestUpdateTrainingConfigServiceLoss:
    def test_persists_submitted_loss(self):
        model = ModelBasic(model_name="my_model")
        db = _db_returning(model)

        _, status = update_training_config_service(
            db, "my_model", _training_config(problem_type_id=ProblemType.REGRESSION, loss=LossFunction.HUBER)
        )

        assert status == 200
        assert model.loss == "huber"

    @pytest.mark.parametrize(
        ("problem_type", "loss"),
        [
            (ProblemType.CLASSIFICATION, LossFunction.MEAN_SQUARED_ERROR),
            (ProblemType.CLASSIFICATION, LossFunction.HUBER),
            (ProblemType.IMAGE_CLASSIFICATION, LossFunction.MEAN_ABSOLUTE_ERROR),
            (ProblemType.REGRESSION, LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY),
            (ProblemType.REGRESSION, LossFunction.CATEGORICAL_CROSSENTROPY),
            (ProblemType.REGRESSION, LossFunction.BINARY_CROSSENTROPY),
        ],
    )
    def test_rejects_a_loss_that_does_not_fit_the_problem_type(self, problem_type, loss):
        model = ModelBasic(model_name="my_model")
        db = _db_returning(model)

        body, status = update_training_config_service(
            db, "my_model", _training_config(problem_type_id=problem_type, loss=loss)
        )

        assert status == 400
        assert loss.value in body["message"]
        assert "Choose one of" in body["message"]
        assert model.loss is None
        db.commit.assert_not_called()

    @pytest.mark.parametrize("problem_type", list(ProblemType))
    def test_every_problem_type_default_is_an_allowed_loss(self, problem_type):
        assert _default_loss(problem_type) in {x.value for x in losses_for_problem_type(problem_type)}

    @pytest.mark.parametrize(
        ("problem_type", "expected"),
        [
            (ProblemType.CLASSIFICATION, "sparse_categorical_crossentropy"),
            (ProblemType.IMAGE_CLASSIFICATION, "sparse_categorical_crossentropy"),
            (ProblemType.REGRESSION, "mean_squared_error"),
        ],
    )
    def test_defaults_from_problem_type_when_loss_omitted(self, problem_type, expected):
        model = ModelBasic(model_name="my_model")
        db = _db_returning(model)

        _, status = update_training_config_service(
            db, "my_model", _training_config(problem_type_id=problem_type, loss=None)
        )

        assert status == 200
        assert model.loss == expected
