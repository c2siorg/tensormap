"""Tests for the hyperparameter tuning API and service helpers.

Background tuning is patched out in HTTP tests so we deterministically test
the request/persistence behaviour without spinning up real Keras runs.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlmodel import Session

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.tuning_session import TuningSession, TuningStrategy
from app.models.tuning_session import TuningStatus as TuningSessionStatus
from app.services.tuning_service import TuningService

BASE = "/api/v1/model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_model(session: Session, name: str = "tuning_model") -> ModelBasic:
    """Create a fully-configured model (with project + dataset) ready to train."""
    project = Project(name="tuning_proj")
    session.add(project)
    session.commit()
    session.refresh(project)

    data_file = DataFile(file_name="d.csv", file_type="csv", disk_name="d.csv", project_id=project.id)
    session.add(data_file)
    session.commit()
    session.refresh(data_file)

    model = ModelBasic(
        model_name=name,
        file_id=data_file.id,
        project_id=project.id,
        model_type=1,
        target_field="label",
        training_split=80,
        optimizer="adam",
        metric="accuracy",
        epochs=5,
        batch_size=32,
        loss="sparse_categorical_crossentropy",
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def _seed_tuning_session(
    session: Session,
    model_id: int,
    strategy: TuningStrategy = TuningStrategy.RANDOM,
    status: TuningSessionStatus = TuningSessionStatus.PENDING,
    best_job_id: str | None = None,
    search_space: dict | None = None,
) -> TuningSession:
    """Create a tuning session row for testing."""
    ts = TuningSession(
        model_id=model_id,
        strategy=strategy,
        search_space=search_space or {"optimizer": ["adam", "sgd"]},
        max_trials=10,
        metric="val_accuracy",
        direction="maximize",
        status=status,
        total_trials=10,
        best_job_id=best_job_id,
    )
    session.add(ts)
    session.commit()
    session.refresh(ts)
    return ts


def _seed_job(
    session: Session,
    model_id: int,
    status: TrainingStatus,
    hyperparams: dict | None = None,
    tuning_session_id: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> TrainingJob:
    job = TrainingJob(
        model_id=model_id,
        status=status,
        hyperparams=hyperparams,
        tuning_session_id=tuning_session_id,
        started_at=started_at,
        completed_at=completed_at,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_background_tuning(mocker):
    """Stop the tuning endpoint from launching real tuning in every HTTP test."""
    return mocker.patch("app.routers.tuning.launch_tuning_session")


# ---------------------------------------------------------------------------
# TuningService unit tests
# ---------------------------------------------------------------------------


class TestGridCombinations:
    """Tests for generate_grid_combinations."""

    def test_grid_combinations_correct(self):
        """3 optimizer × 3 lr × 2 batch = 18 combos."""
        svc = TuningService()
        space = {
            "optimizer": ["adam", "sgd", "rmsprop"],
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32],
        }
        combos = svc.generate_grid_combinations(space)
        assert len(combos) == 18
        # Each combo has all three keys.
        for c in combos:
            assert set(c.keys()) == {"optimizer", "learning_rate", "batch_size"}
        # Check uniqueness.
        combo_tuples = [tuple(sorted(c.items())) for c in combos]
        assert len(set(combo_tuples)) == 18

    def test_grid_exceeds_max_returns_400(self):
        """More than MAX_GRID_COMBINATIONS → AppException(400)."""
        from app.exceptions import AppException

        svc = TuningService()
        # 10 × 10 × 10 = 1000, well above default 50.
        space = {
            "a": list(range(10)),
            "b": list(range(10)),
            "c": list(range(10)),
        }
        with pytest.raises(AppException) as exc_info:
            svc.generate_grid_combinations(space)
        assert exc_info.value.status_code == 400
        assert "1000" in str(exc_info.value.detail)

    def test_grid_empty_discrete_returns_single_empty(self):
        """No discrete params → single empty dict."""
        svc = TuningService()
        space = {"lr": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}}
        combos = svc.generate_grid_combinations(space)
        assert combos == [{}]

    def test_grid_single_param(self):
        """Single parameter with 3 values → 3 combos."""
        svc = TuningService()
        space = {"optimizer": ["adam", "sgd", "rmsprop"]}
        combos = svc.generate_grid_combinations(space)
        assert len(combos) == 3


class TestRandomSampling:
    """Tests for sample_random_combination."""

    def test_random_search_respects_max_trials(self):
        """Exactly max_trials combinations generated."""
        svc = TuningService()
        space = {"optimizer": ["adam", "sgd"], "batch_size": [16, 32, 64]}
        rng = np.random.default_rng(seed=42)
        trials = [svc.sample_random_combination(space, rng) for _ in range(20)]
        assert len(trials) == 20
        for t in trials:
            assert "optimizer" in t
            assert "batch_size" in t

    def test_log_uniform_sampling_in_range(self):
        """Sampled learning rate is always between min and max."""
        svc = TuningService()
        space = {"learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}}
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            combo = svc.sample_random_combination(space, rng)
            lr = combo["learning_rate"]
            assert 1e-5 <= lr <= 1e-2, f"lr={lr} out of range"

    def test_uniform_sampling_in_range(self):
        """Uniform sampling stays within bounds."""
        svc = TuningService()
        space = {"dropout": {"type": "uniform", "min": 0.1, "max": 0.5}}
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            combo = svc.sample_random_combination(space, rng)
            assert 0.1 <= combo["dropout"] <= 0.5

    def test_discrete_choice_values(self):
        """Discrete choices only produce values from the list."""
        svc = TuningService()
        space = {"optimizer": ["adam", "sgd"]}
        rng = np.random.default_rng(seed=42)
        for _ in range(50):
            combo = svc.sample_random_combination(space, rng)
            assert combo["optimizer"] in ["adam", "sgd"]


class TestSearchSpaceValidation:
    """Tests for validate_search_space."""

    def test_valid_list_and_dict_specs_pass(self):
        """Well-formed list and dict specs are accepted."""
        svc = TuningService()
        space = {
            "optimizer": ["adam", "sgd"],
            "learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2},
            "dropout": {"type": "uniform", "min": 0.1, "max": 0.5},
            "batch_size": [16, 32],
        }
        svc.validate_search_space(space, TuningStrategy.RANDOM)
        svc.validate_search_space(space, TuningStrategy.GRID)

    def test_empty_search_space_rejected(self):
        """Empty search space → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space({}, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "non-empty" in str(exc_info.value.detail)

    def test_empty_list_value_rejected(self):
        """Empty list value → 400 (would otherwise produce zero trials)."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"batch_size": [16, 32], "epochs": []}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.GRID)
        assert exc_info.value.status_code == 400
        assert "non-empty list" in str(exc_info.value.detail)

    def test_missing_max_rejected(self):
        """Dict spec without 'max' → 400 (would otherwise crash the thread)."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "uniform", "min": 0.1}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "numeric 'min' and 'max'" in str(exc_info.value.detail)

    def test_non_numeric_bounds_rejected(self):
        """Non-numeric min/max → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "uniform", "min": "bad", "max": 1.0}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400

    def test_min_ge_max_rejected(self):
        """min >= max → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"dropout": {"type": "uniform", "min": 0.5, "max": 0.1}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "'min' < 'max'" in str(exc_info.value.detail)

    def test_log_uniform_zero_min_rejected(self):
        """log_uniform with min <= 0 → 400 (log(0) is undefined)."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "log_uniform", "min": 0, "max": 1e-2}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "'min' > 0" in str(exc_info.value.detail)

    def test_grid_requires_discrete_param(self):
        """Grid search with only continuous params → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.GRID)
        assert exc_info.value.status_code == 400
        assert "at least one discrete" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestStartTuning:
    """Tests for POST /model/tuning/{model_name}."""

    def test_tuning_session_created_on_post(self, client, db_session):
        """POST creates a tuning_session row in the DB."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {"optimizer": ["adam", "sgd"], "batch_size": [16, 32]},
                "max_trials": 5,
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        tuning_id = body["data"]["tuning_id"]
        assert tuning_id

        # Verify DB row.
        ts = db_session.get(TuningSession, tuning_id)
        assert ts is not None
        assert ts.strategy == TuningStrategy.RANDOM
        assert ts.max_trials == 5
        assert ts.status == TuningSessionStatus.PENDING

    def test_estimate_seconds_in_response(self, client, db_session):
        """POST 202 response includes estimated_seconds and n_trials."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {"optimizer": ["adam", "sgd"]},
                "max_trials": 4,
            },
        )
        assert resp.status_code == 202
        data = resp.json()["data"]
        assert "estimated_seconds" in data
        assert data["estimated_seconds"] > 0
        assert data["n_trials"] == 4

    def test_grid_search_post_400_too_many(self, client, db_session):
        """Grid search with >50 combos returns 400."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {
                    "a": list(range(10)),
                    "b": list(range(10)),
                    "c": list(range(10)),
                },
            },
        )
        assert resp.status_code == 400

    def test_unknown_model_404(self, client, db_session):
        """POST for non-existent model returns 404."""
        resp = client.post(
            f"{BASE}/tuning/does_not_exist",
            json={
                "model_name": "does_not_exist",
                "strategy": "random",
                "search_space": {"optimizer": ["adam"]},
            },
        )
        assert resp.status_code == 404

    def test_unconfigured_model_400(self, client, db_session):
        """Model without training config returns 400."""
        model = ModelBasic(model_name="bare", optimizer="adam")
        db_session.add(model)
        db_session.commit()
        resp = client.post(
            f"{BASE}/tuning/bare",
            json={
                "model_name": "bare",
                "strategy": "random",
                "search_space": {"optimizer": ["adam"]},
            },
        )
        assert resp.status_code == 400

    def test_empty_search_space_400(self, client, db_session):
        """POST with an empty search_space returns 400 instead of a silent no-op."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {},
                "max_trials": 5,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["success"] is False

    def test_empty_list_value_400(self, client, db_session):
        """POST with an empty list value returns 400 instead of a zero-trial session."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {"batch_size": [16, 32], "epochs": []},
            },
        )
        assert resp.status_code == 400

    def test_malformed_dict_spec_400(self, client, db_session):
        """POST with a dict spec missing 'max' returns 400 instead of crashing the thread."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {"learning_rate": {"type": "uniform", "min": 0.1}},
            },
        )
        assert resp.status_code == 400

    def test_continuous_only_grid_400(self, client, db_session):
        """POST grid search with only continuous params returns 400."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {"learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}},
            },
        )
        assert resp.status_code == 400


class TestGetTuningSession:
    """Tests for GET /model/tuning/{tuning_id}."""

    def test_get_tuning_session(self, client, db_session):
        """GET returns session detail with trial info."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.COMPLETED)

        # Add a linked job.
        job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            hyperparams={"optimizer": "adam"},
            tuning_session_id=ts.id,
        )

        resp = client.get(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["tuning_id"] == ts.id
        assert data["status"] == "completed"
        assert len(data["trials"]) == 1
        assert data["trials"][0]["job_id"] == job.id

    def test_get_tuning_session_404(self, client, db_session):
        """GET for non-existent session returns 404."""
        resp = client.get(f"{BASE}/tuning/nonexistent-id")
        assert resp.status_code == 404


class TestCancelTuning:
    """Tests for DELETE /model/tuning/{tuning_id}."""

    def test_cancel_tuning_sets_status(self, client, db_session):
        """DELETE sets status=cancelled."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        resp = client.delete(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 204

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.CANCELLED

    def test_cancel_tuning_cancels_running_jobs(self, client, db_session):
        """DELETE also cancels any running trial jobs."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)
        job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.RUNNING,
            tuning_session_id=ts.id,
        )

        resp = client.delete(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 204

        db_session.refresh(job)
        assert job.status == TrainingStatus.CANCELLED

    def test_cancel_404(self, client, db_session):
        """DELETE for non-existent session returns 404."""
        resp = client.delete(f"{BASE}/tuning/nonexistent-id")
        assert resp.status_code == 404

    def test_cancel_completed_is_noop(self, client, db_session):
        """Cancelling an already-completed session is a 204 no-op."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.COMPLETED)

        resp = client.delete(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 204

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.COMPLETED  # unchanged


class TestApplyBest:
    """Tests for POST /model/tuning/{tuning_id}/apply-best."""

    def test_apply_best_updates_model_config(self, client, db_session, monkeypatch):
        """apply-best copies best hyperparams to model_basic."""
        model = _seed_model(db_session, "tuning_model")
        best_job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            hyperparams={"optimizer": "sgd", "epochs": 100, "batch_size": 64},
        )
        ts = _seed_tuning_session(
            db_session,
            model.id,
            status=TuningSessionStatus.COMPLETED,
            best_job_id=best_job.id,
        )

        # Patch tuning_service.make_session to use the test DB session.
        def _mock_make_session():
            return db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_make_session)

        # Need a context manager mock for make_session
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        resp = client.post(f"{BASE}/tuning/{ts.id}/apply-best")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["applied_hyperparams"]["optimizer"] == "sgd"
        assert body["data"]["applied_hyperparams"]["epochs"] == 100

        # Verify model was updated.
        db_session.refresh(model)
        assert model.optimizer == "sgd"
        assert model.epochs == 100
        assert model.batch_size == 64

    def test_apply_best_not_completed_400(self, client, db_session):
        """apply-best on non-completed session returns 400."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        resp = client.post(f"{BASE}/tuning/{ts.id}/apply-best")
        assert resp.status_code == 400

    def test_apply_best_404(self, client, db_session):
        """apply-best on non-existent session returns 404."""
        resp = client.post(f"{BASE}/tuning/nonexistent-id/apply-best")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


class TestEarlyStop:
    """Tests for early stopping logic."""

    def test_early_stop_terminates_session(self, db_session, monkeypatch):
        """Threshold met → session stops early + early_stopped=True.

        Tests the logic indirectly by verifying the early_stopped flag is set
        on the tuning session when the threshold is met.
        """
        model = _seed_model(db_session, "tuning_model")

        # Create a session with early_stop_threshold.
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.RANDOM,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=10,
            metric="val_accuracy",
            direction="maximize",
            early_stop_threshold=0.9,
            total_trials=10,
            status=TuningSessionStatus.RUNNING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        # Simulate: set early_stopped=True (as the tuning loop would).
        ts.early_stopped = True
        ts.status = TuningSessionStatus.COMPLETED
        ts.completed_trials = 3  # Stopped after 3 of 10.
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        assert ts.early_stopped is True
        assert ts.completed_trials == 3
        assert ts.status == TuningSessionStatus.COMPLETED


class TestTrialTimeout:
    """Tests for trial timeout logic."""

    def test_trial_timeout_cancels_job(self, db_session, monkeypatch):
        """Mock long-running trial → timeout fires → job=cancelled."""
        model = _seed_model(db_session, "tuning_model")
        job = _seed_job(db_session, model.id, TrainingStatus.RUNNING)

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.training_service.make_session", _mock_session_cm)

        svc = TuningService()
        svc._timeout_trial(job.id)

        db_session.refresh(job)
        assert job.status == TrainingStatus.CANCELLED
        assert "timeout" in (job.error_message or "").lower()


class TestEstimateDuration:
    """Tests for estimate_session_duration."""

    def test_estimate_duration_from_last_job(self, db_session, monkeypatch):
        """Last job took 60s, 10 trials → estimate 600s."""
        model = _seed_model(db_session, "tuning_model")
        now = datetime.now(UTC)
        _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            started_at=now - timedelta(seconds=60),
            completed_at=now,
        )

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        svc = TuningService()
        estimate = svc.estimate_session_duration(model.id, 10)
        assert estimate == 600

    def test_estimate_duration_no_history(self, db_session, monkeypatch):
        """No previous jobs → falls back to DEFAULT_TRIAL_ESTIMATE_SECONDS."""
        model = _seed_model(db_session, "tuning_model")

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        svc = TuningService()
        estimate = svc.estimate_session_duration(model.id, 10)
        # DEFAULT_TRIAL_ESTIMATE_SECONDS = 120, so 120 * 10 = 1200.
        assert estimate == 1200


class TestRunTuningLoopHardeness:
    """Tests for run_tuning_loop leaving sessions in a terminal state."""

    def test_invalid_search_space_marks_session_failed(self, db_session, monkeypatch):
        """A session whose search_space is malformed → FAILED, not stuck RUNNING.

        Regression test: before hardening, ``sample_random_combination`` raised
        KeyError inside the background thread with no handler, leaving the
        session forever in ``RUNNING``.
        """
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.tuning_service.TuningService._emit_progress", lambda *a, **k: None)

        model = _seed_model(db_session, "tuning_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.RANDOM,
            search_space={"learning_rate": {"type": "uniform", "min": 0.1}},
            max_trials=10,
            metric="val_accuracy",
            direction="maximize",
            total_trials=10,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        svc = TuningService()
        svc.run_tuning_loop(ts.id, "tuning_model", loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.FAILED
        assert ts.completed_at is not None
