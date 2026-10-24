"""Comprehensive unit and integration tests for tuning session lifecycle, concurrency,
orphan recovery, guaranteed model restoration, cancellation socket emissions,
and tabular validation data logging during training.
"""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.exceptions import AppException
from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.tuning_session import TuningSession, TuningStatus, TuningStrategy
from app.services.model_run import _run
from app.services.tuning_service import (
    MAX_CONCURRENT_TUNING_SESSIONS,
    TuningService,
    check_tuning_concurrency_limit,
    emit_tuning_cancellation,
    orphan_recovery_tuning,
)
from app.shared.enums import ProblemType

BASE_URL = "/api/v1/model"


# ---------------------------------------------------------------------------
# Fixtures & Seed Helpers
# ---------------------------------------------------------------------------


def _seed_project_and_file(session: Session, file_name: str = "dataset.csv") -> tuple[Project, DataFile]:
    project = Project(name=f"proj_{file_name}")
    session.add(project)
    session.commit()
    session.refresh(project)

    data_file = DataFile(
        file_name=file_name,
        file_type="csv",
        disk_name=file_name,
        project_id=project.id,
    )
    session.add(data_file)
    session.commit()
    session.refresh(data_file)
    return project, data_file


def _seed_model(
    session: Session,
    name: str = "lifecycle_model",
    problem_type: int = ProblemType.CLASSIFICATION,
    training_split: float = 80.0,
    epochs: int = 5,
    batch_size: int = 32,
    optimizer: str = "adam",
) -> ModelBasic:
    project, data_file = _seed_project_and_file(session, f"{name}.csv")
    model = ModelBasic(
        model_name=name,
        file_id=data_file.id,
        project_id=project.id,
        model_type=problem_type,
        target_field="target",
        training_split=training_split,
        optimizer=optimizer,
        metric="accuracy" if problem_type == ProblemType.CLASSIFICATION else "mse",
        epochs=epochs,
        batch_size=batch_size,
        loss="sparse_categorical_crossentropy" if problem_type == ProblemType.CLASSIFICATION else "mean_squared_error",
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def _seed_tuning_session(
    session: Session,
    model_id: int,
    status: TuningStatus = TuningStatus.PENDING,
    strategy: TuningStrategy = TuningStrategy.RANDOM,
    total_trials: int = 5,
) -> TuningSession:
    ts = TuningSession(
        model_id=model_id,
        strategy=strategy,
        search_space={"optimizer": ["adam", "sgd"]},
        max_trials=total_trials,
        metric="val_accuracy",
        direction="maximize",
        status=status,
        total_trials=total_trials,
    )
    session.add(ts)
    session.commit()
    session.refresh(ts)
    return ts


def _seed_job(
    session: Session,
    model_id: int,
    status: TrainingStatus = TrainingStatus.RUNNING,
    tuning_session_id: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> TrainingJob:
    job = TrainingJob(
        model_id=model_id,
        status=status,
        tuning_session_id=tuning_session_id,
        started_at=started_at or datetime.now(UTC),
        completed_at=completed_at,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


# ===========================================================================
# 1. Tabular Model Training Validation Data Tests
# ===========================================================================


class TestTabularValidationData:
    """Verifies that model_run passes validation_data to model.fit for tabular datasets."""

    def test_tabular_classification_passes_validation_data_to_fit(self, db_session):
        model = _seed_model(db_session, "tabular_clf_model", ProblemType.CLASSIFICATION, training_split=80.0)

        mock_keras_model = MagicMock()
        mock_features = MagicMock()

        dummy_x_train = MagicMock(name="x_train")
        dummy_y_train = MagicMock(name="y_train")
        dummy_x_test = ["sample1", "sample2"]  # len > 0
        dummy_y_test = [0, 1]

        with (
            patch("app.services.model_run._build_training_callbacks", return_value=[]),
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/file.csv"),
            patch("app.services.model_run.pd.read_csv", return_value=mock_features),
            patch(
                "app.services.model_run._prepare_training_data",
                return_value=(dummy_x_train, dummy_y_train, dummy_x_test, dummy_y_test),
            ),
            patch("builtins.open", MagicMock()),
            patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_keras_model),
            patch("app.services.model_run._emits_logits", return_value=False),
        ):
            _run(model.model_name, db_session)

        # Assert validation_data was supplied to model.fit
        mock_keras_model.fit.assert_called_once()
        fit_kwargs = mock_keras_model.fit.call_args.kwargs
        assert "validation_data" in fit_kwargs
        assert fit_kwargs["validation_data"] == (dummy_x_test, dummy_y_test)

        # Assert model.evaluate was called on the test split
        mock_keras_model.evaluate.assert_called_once_with(dummy_x_test, dummy_y_test, callbacks=[], verbose=0)

    def test_tabular_regression_passes_validation_data_to_fit(self, db_session):
        model = _seed_model(db_session, "tabular_reg_model", ProblemType.REGRESSION, training_split=75.0)

        mock_keras_model = MagicMock()
        mock_features = MagicMock()

        dummy_x_train = MagicMock(name="x_train_reg")
        dummy_y_train = MagicMock(name="y_train_reg")
        dummy_x_test = ["sampleA", "sampleB"]
        dummy_y_test = [12.5, 18.2]

        with (
            patch("app.services.model_run._build_training_callbacks", return_value=[]),
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/file.csv"),
            patch("app.services.model_run.pd.read_csv", return_value=mock_features),
            patch(
                "app.services.model_run._prepare_training_data",
                return_value=(dummy_x_train, dummy_y_train, dummy_x_test, dummy_y_test),
            ),
            patch("builtins.open", MagicMock()),
            patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_keras_model),
            patch("app.services.model_run._emits_logits", return_value=False),
        ):
            _run(model.model_name, db_session)

        mock_keras_model.fit.assert_called_once()
        fit_kwargs = mock_keras_model.fit.call_args.kwargs
        assert fit_kwargs["validation_data"] == (dummy_x_test, dummy_y_test)

    def test_tabular_empty_test_split_handles_gracefully(self, db_session):
        """When test split array is empty, validation_data is None and evaluate is skipped."""
        model = _seed_model(db_session, "split_100_model", ProblemType.CLASSIFICATION, training_split=80.0)

        mock_keras_model = MagicMock()
        mock_features = MagicMock()

        dummy_x_train = MagicMock(name="x_train_full")
        dummy_y_train = MagicMock(name="y_train_full")
        empty_x_test = []
        empty_y_test = []

        with (
            patch("app.services.model_run._build_training_callbacks", return_value=[]),
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/file.csv"),
            patch("app.services.model_run.pd.read_csv", return_value=mock_features),
            patch(
                "app.services.model_run._prepare_training_data",
                return_value=(dummy_x_train, dummy_y_train, empty_x_test, empty_y_test),
            ),
            patch("builtins.open", MagicMock()),
            patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_keras_model),
            patch("app.services.model_run._emits_logits", return_value=False),
        ):
            _run(model.model_name, db_session)

        fit_kwargs = mock_keras_model.fit.call_args.kwargs
        assert fit_kwargs["validation_data"] is None
        mock_keras_model.evaluate.assert_not_called()


# ===========================================================================
# 2. Tuning Session Concurrency Limit Tests
# ===========================================================================


class TestTuningConcurrencyLimit:
    """Verifies that concurrent tuning sessions are bounded and return 429 when max is reached."""

    def test_check_concurrency_limit_passes_when_under_limit(self, db_session):
        model = _seed_model(db_session, "concurrency_model_1")
        # 1 active session (limit is 2)
        _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)

        # Should not raise
        check_tuning_concurrency_limit(db_session)

    def test_check_concurrency_limit_raises_429_when_max_reached(self, db_session):
        model = _seed_model(db_session, "concurrency_model_2")
        for _ in range(MAX_CONCURRENT_TUNING_SESSIONS):
            _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)

        with pytest.raises(AppException) as exc:
            check_tuning_concurrency_limit(db_session)

        assert exc.value.status_code == 429
        assert "Max concurrent tuning sessions reached" in exc.value.detail

    def test_completed_and_cancelled_sessions_do_not_count_towards_limit(self, db_session):
        model = _seed_model(db_session, "concurrency_model_3")
        _seed_tuning_session(db_session, model.id, status=TuningStatus.COMPLETED)
        _seed_tuning_session(db_session, model.id, status=TuningStatus.CANCELLED)
        _seed_tuning_session(db_session, model.id, status=TuningStatus.FAILED)

        # None of these are PENDING or RUNNING
        check_tuning_concurrency_limit(db_session)

    def test_start_tuning_endpoint_returns_429_when_max_reached(self, client, db_session):
        model = _seed_model(db_session, "api_concurrency_model")
        for _ in range(MAX_CONCURRENT_TUNING_SESSIONS):
            _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)

        resp = client.post(
            f"{BASE_URL}/tuning/{model.model_name}",
            json={
                "model_name": model.model_name,
                "strategy": "random",
                "search_space": {"optimizer": ["adam", "sgd"]},
                "max_trials": 5,
            },
        )
        assert resp.status_code == 429
        assert "Max concurrent tuning sessions reached" in resp.json()["message"]


# ===========================================================================
# 3. Tuning Session Orphan Recovery Tests
# ===========================================================================


class TestTuningOrphanRecovery:
    """Verifies that orphaned tuning sessions and their jobs are recovered on startup."""

    def test_recovers_pending_and_running_tuning_sessions(self, db_session):
        model = _seed_model(db_session, "orphan_model")
        ts_pending = _seed_tuning_session(db_session, model.id, status=TuningStatus.PENDING)
        ts_running = _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)

        job_pending = _seed_job(db_session, model.id, TrainingStatus.PENDING, tuning_session_id=ts_running.id)
        job_running = _seed_job(db_session, model.id, TrainingStatus.RUNNING, tuning_session_id=ts_running.id)

        recovered_count = orphan_recovery_tuning(db_session)
        assert recovered_count == 2

        db_session.refresh(ts_pending)
        db_session.refresh(ts_running)
        assert ts_pending.status == TuningStatus.FAILED
        assert ts_pending.completed_at is not None
        assert ts_running.status == TuningStatus.FAILED
        assert ts_running.completed_at is not None

        db_session.refresh(job_pending)
        db_session.refresh(job_running)
        assert job_pending.status == TrainingStatus.FAILED
        assert job_pending.error_message == "Recovered orphaned tuning trial after restart"
        assert job_running.status == TrainingStatus.FAILED
        assert job_running.error_message == "Recovered orphaned tuning trial after restart"

    def test_does_not_modify_completed_or_cancelled_sessions(self, db_session):
        model = _seed_model(db_session, "finished_model")
        ts_completed = _seed_tuning_session(db_session, model.id, status=TuningStatus.COMPLETED)
        ts_cancelled = _seed_tuning_session(db_session, model.id, status=TuningStatus.CANCELLED)

        recovered_count = orphan_recovery_tuning(db_session)
        assert recovered_count == 0

        db_session.refresh(ts_completed)
        db_session.refresh(ts_cancelled)
        assert ts_completed.status == TuningStatus.COMPLETED
        assert ts_cancelled.status == TuningStatus.CANCELLED

    def test_orphan_recovery_tuning_is_idempotent(self, db_session):
        model = _seed_model(db_session, "idempotent_model")
        _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)

        first_pass = orphan_recovery_tuning(db_session)
        assert first_pass == 1

        second_pass = orphan_recovery_tuning(db_session)
        assert second_pass == 0

    def test_orphan_recovery_with_own_session(self, db_session, monkeypatch):
        monkeypatch.setattr("app.services.tuning_service.make_session", lambda: Session(db_session.get_bind()))

        model = _seed_model(db_session, "own_session_orphan_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)

        recovered = orphan_recovery_tuning()
        assert recovered == 1

        db_session.refresh(ts)
        assert ts.status == TuningStatus.FAILED


# ===========================================================================
# 4. Guaranteed Model Config Restoration Tests
# ===========================================================================


class TestTuningModelConfigRestoration:
    """Verifies that ModelBasic is restored even when trials fail, throw exceptions, or timeout."""

    def test_model_config_restored_when_model_run_raises(self, db_session, monkeypatch):
        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr(
            "app.services.model_run.model_run",
            MagicMock(side_effect=RuntimeError("Keras graph crashed")),
        )

        model = _seed_model(db_session, "crash_resilient_model", epochs=5, batch_size=32, optimizer="adam")
        ts = _seed_tuning_session(db_session, model.id)

        svc = TuningService()
        trial_params = {"optimizer": "sgd", "epochs": 20, "batch_size": 64}

        svc._run_single_trial(
            session_id=ts.id,
            model_name=model.model_name,
            hyperparams=trial_params,
            trial_num=1,
            loop=None,
        )

        db_session.refresh(model)
        assert model.optimizer == "adam"
        assert model.epochs == 5
        assert model.batch_size == 32

    def test_model_config_restored_when_status_update_fails(self, db_session, monkeypatch):
        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.model_run.model_run", MagicMock(side_effect=ValueError("Data error")))
        monkeypatch.setattr(
            "app.services.training_service.update_job_status",
            MagicMock(side_effect=RuntimeError("DB write failed")),
        )

        model = _seed_model(db_session, "db_fail_model", epochs=10, batch_size=16, optimizer="rmsprop")
        ts = _seed_tuning_session(db_session, model.id)

        svc = TuningService()
        trial_params = {"optimizer": "adam", "epochs": 50, "batch_size": 128}

        with pytest.raises(RuntimeError, match="DB write failed"):
            svc._run_single_trial(
                session_id=ts.id,
                model_name=model.model_name,
                hyperparams=trial_params,
                trial_num=1,
                loop=None,
            )

        db_session.refresh(model)
        assert model.optimizer == "rmsprop"
        assert model.epochs == 10
        assert model.batch_size == 16


# ===========================================================================
# 5. Tuning Cancellation Lifecycle & Socket Emission Tests
# ===========================================================================


class TestTuningCancellationLifecycle:
    """Verifies that cancel_tuning stamps timestamps and emits Socket.IO cancellation event."""

    def test_cancel_tuning_stamps_completed_at_and_jobs(self, client, db_session):
        model = _seed_model(db_session, "cancel_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningStatus.RUNNING)
        job = _seed_job(db_session, model.id, TrainingStatus.RUNNING, tuning_session_id=ts.id)

        with patch("app.routers.tuning.emit_tuning_cancellation") as mock_emit:
            resp = client.delete(f"{BASE_URL}/tuning/{ts.id}")
            assert resp.status_code == 204
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == ts.id

        db_session.refresh(ts)
        assert ts.status == TuningStatus.CANCELLED
        assert ts.completed_at is not None

        db_session.refresh(job)
        assert job.status == TrainingStatus.CANCELLED
        assert job.completed_at is not None

    def test_emit_tuning_cancellation_emits_to_room(self):
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        with (
            patch("app.services.tuning_service.sio") as mock_sio,
            patch("asyncio.run_coroutine_threadsafe") as mock_run,
        ):
            mock_future = MagicMock()
            mock_run.return_value = mock_future

            emit_tuning_cancellation("session-uuid-123", mock_loop)

            mock_run.assert_called_once()
            mock_sio.emit.assert_called_once()
            call_args = mock_sio.emit.call_args
            event_name = call_args[0][0]
            event_payload = call_args[0][1]
            kwargs = call_args[1]

            assert event_name == "tuning_progress"
            assert event_payload["type"] == "tuning_complete"
            assert event_payload["status"] == "cancelled"
            assert kwargs["room"] == "tuning:session-uuid-123"

    def test_cancel_already_cancelled_session_returns_204(self, client, db_session):
        model = _seed_model(db_session, "already_cancelled_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningStatus.CANCELLED)

        resp = client.delete(f"{BASE_URL}/tuning/{ts.id}")
        assert resp.status_code == 204


# ===========================================================================
# 6. Defensive Duration Estimation Bounds Tests
# ===========================================================================


class TestDurationEstimationDefensiveBounds:
    """Verifies that duration estimation is bounded against clock skew and negative timestamps."""

    def test_duration_estimation_guards_against_clock_skew(self, db_session, monkeypatch):
        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        model = _seed_model(db_session, "clock_skew_model")
        # Clock skew: completed_at is before started_at
        now = datetime.now(UTC)
        _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            started_at=now,
            completed_at=now - timedelta(seconds=120),
        )

        svc = TuningService()
        est = svc.estimate_session_duration(model.id, n_trials=5)
        # Should be clamped to at least 1 second per trial => >= 5
        assert est >= 5

    def test_duration_estimation_zero_elapsed_clamped_to_minimum(self, db_session, monkeypatch):
        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        model = _seed_model(db_session, "zero_elapsed_model")
        now = datetime.now(UTC)
        _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            started_at=now,
            completed_at=now,  # 0 elapsed
        )

        svc = TuningService()
        est = svc.estimate_session_duration(model.id, n_trials=10)
        assert est >= 10

    def test_duration_estimation_fallback_when_no_completed_jobs(self, db_session, monkeypatch):
        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        model = _seed_model(db_session, "no_history_model")
        svc = TuningService()
        est = svc.estimate_session_duration(model.id, n_trials=3)
        # Default is 120 * 3 = 360
        assert est == 360
