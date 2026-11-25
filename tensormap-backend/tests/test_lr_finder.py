"""Tests for learning rate range test profiler and optimal lr finder."""

from unittest.mock import MagicMock, patch

import pytest
import tensorflow as tf

from app.callbacks.lr_finder import LRFinderCallback, _extract_lr
from app.exceptions import AppException
from app.models.ml import ModelBasic
from app.services.lr_finder import find_optimal_learning_rate


class TestLRFinderCallback:
    def test_multiplier_calculation(self):
        cb = LRFinderCallback(start_lr=1e-5, end_lr=1e-1, num_batches=100)
        expected_mult = (1e-1 / 1e-5) ** (1.0 / 100)
        assert abs(cb.lr_multiplier - expected_mult) < 1e-9

    def test_callback_records_history_and_updates_lr(self):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse")

        cb = LRFinderCallback(start_lr=1e-4, end_lr=1e-1, num_batches=10)
        cb.set_model(model)

        cb.on_train_begin()
        assert _extract_lr(model.optimizer) == pytest.approx(1e-4)

        # Simulate 5 batch ends
        for i in range(5):
            cb.on_train_batch_end(i, logs={"loss": 1.0 - (i * 0.1)})

        assert len(cb.history) == 5
        assert cb.diverged is False

        # End training: initial lr and weights restored
        cb.on_train_end()
        assert _extract_lr(model.optimizer) == pytest.approx(0.01)

    def test_callback_stops_early_on_divergence(self):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse")

        cb = LRFinderCallback(start_lr=1e-3, end_lr=1.0, num_batches=50, stop_factor=3.0)
        cb.set_model(model)
        cb.on_train_begin()

        # Batch 1: low loss
        cb.on_train_batch_end(0, logs={"loss": 0.5})
        # Batch 2: explosion > 3 * 0.5
        cb.on_train_batch_end(1, logs={"loss": 20.0})

        assert cb.diverged is True
        assert getattr(model, "stop_training", False) is True

    def test_callback_computes_suggested_lr(self):
        cb = LRFinderCallback()
        # Synthetic curve: loss descends from 2.0 to 0.2, then rises to 5.0
        synthetic_lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        synthetic_losses = [2.0, 1.8, 1.2, 0.4, 0.2, 3.5]
        cb.history = [
            {"learning_rate": lr, "loss": loss} for lr, loss in zip(synthetic_lrs, synthetic_losses, strict=True)
        ]
        cb.batch_count = len(synthetic_lrs)

        results = cb.get_results()
        assert results["suggested_lr"] is not None
        # Steepest slope happens between 1e-3 and 1e-2
        assert 1e-4 <= results["suggested_lr"] <= 1e-1
        assert results["min_loss"] == pytest.approx(0.2)
        assert results["diverged"] is False

    def test_get_results_empty_history(self):
        cb = LRFinderCallback()
        results = cb.get_results()
        assert results["history"] == []
        assert results["suggested_lr"] is None
        assert results["min_loss"] is None
        assert results["num_batches_completed"] == 0


class TestLRFinderService:
    def test_find_optimal_learning_rate_model_not_found(self):
        db = MagicMock()
        db.exec.return_value.first.return_value = None

        with pytest.raises(AppException) as exc_info:
            find_optimal_learning_rate("nonexistent", db)
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.message

    def test_find_optimal_learning_rate_missing_file_id(self):
        db = MagicMock()
        model_config = MagicMock(spec=ModelBasic)
        model_config.file_id = None
        db.exec.return_value.first.return_value = model_config

        with pytest.raises(AppException) as exc_info:
            find_optimal_learning_rate("test_model", db)
        assert exc_info.value.status_code == 400
        assert "Dataset not linked" in exc_info.value.message

    def test_find_optimal_learning_rate_end_to_end(self, tmp_path):
        db = MagicMock()
        model_config = MagicMock(spec=ModelBasic)
        model_config.file_id = "fake-file-id"
        model_config.model_name = "test_model"
        model_config.target_field = "label"
        model_config.training_split = 80.0
        model_config.optimizer = "adam"
        model_config.loss = "sparse_categorical_crossentropy"
        model_config.metric = "accuracy"
        model_config.batch_size = 16
        db.exec.return_value.first.return_value = model_config

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x1,x2,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")

        mock_keras_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(2)])
        json_path = tmp_path / "model.json"
        json_path.write_text(mock_keras_model.to_json(), encoding="utf-8")

        with (
            patch("app.services.lr_finder._helper_generate_file_location", return_value=str(csv_path)),
            patch("app.services.lr_finder._helper_generate_json_model_file_location", return_value=str(json_path)),
        ):
            results = find_optimal_learning_rate("test_model", db, start_lr=1e-4, end_lr=1e-1, num_batches=10)

        assert "history" in results
        assert "suggested_lr" in results
        assert results["num_batches_completed"] > 0


class TestLRFinderRouter:
    def test_lr_find_endpoint(self, client):
        payload = {
            "model_name": "test_model",
            "start_lr": 1e-5,
            "end_lr": 0.1,
            "num_batches": 20,
        }
        mock_results = {
            "history": [{"learning_rate": 1e-5, "loss": 1.2}],
            "suggested_lr": 1e-4,
            "min_loss": 0.8,
            "diverged": False,
            "num_batches_completed": 20,
        }

        with patch("app.routers.training.find_optimal_learning_rate", return_value=mock_results):
            response = client.post("/api/v1/model/lr-find", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["suggested_lr"] == 1e-4
        assert body["data"]["num_batches_completed"] == 20
