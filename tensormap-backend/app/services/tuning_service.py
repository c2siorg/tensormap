"""Hyperparameter tuning service — grid search and random search.

Trials execute SEQUENTIALLY in a background thread to avoid thread pool
exhaustion.  Grid search is capped at ``MAX_GRID_COMBINATIONS`` to prevent
combinatorial explosion.  Each trial has a hard wall-clock limit enforced by
``threading.Timer`` that marks the job CANCELLED in the DB so the existing
``CancellationCheckCallback`` stops the Keras loop.

Progress is emitted via Socket.IO to a ``tuning:{session_id}`` room.
"""

import asyncio
import itertools
import os
import threading
from datetime import UTC, datetime

import numpy as np
from sqlmodel import Session, select

from app.database import engine
from app.exceptions import AppException
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.models.tuning_session import TuningSession, TuningStatus, TuningStrategy
from app.shared.constants import SOCKETIO_DL_NAMESPACE
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)

MAX_GRID_COMBINATIONS = int(os.getenv("MAX_GRID_COMBINATIONS", "50"))
TRIAL_TIMEOUT_SECONDS = int(os.getenv("TRIAL_TIMEOUT_SECONDS", "600"))
DEFAULT_TRIAL_ESTIMATE_SECONDS = int(os.getenv("DEFAULT_TRIAL_ESTIMATE_SECONDS", "120"))

# Strong references to in-flight tuning tasks so asyncio doesn't GC them.
_tuning_tasks: set[asyncio.Task] = set()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def make_session() -> Session:
    """Create a new DB session bound to the app engine.

    Used by the tuning loop running in a background thread.
    """
    return Session(engine)


class TuningService:
    """Orchestrates hyperparameter tuning sessions with grid or random search."""

    # ------------------------------------------------------------------
    # Combination generators
    # ------------------------------------------------------------------

    def generate_grid_combinations(self, search_space: dict) -> list[dict]:
        """Generate all Cartesian-product combinations for discrete parameters.

        Raises ``AppException(400)`` if the number of combinations exceeds
        ``MAX_GRID_COMBINATIONS``.
        """
        discrete_params = {k: v for k, v in search_space.items() if isinstance(v, list)}
        if not discrete_params:
            return [{}]
        keys = list(discrete_params.keys())
        values = [discrete_params[k] for k in keys]
        combinations = list(itertools.product(*values))
        if len(combinations) > MAX_GRID_COMBINATIONS:
            raise AppException(
                400,
                f"Grid search would produce {len(combinations)} combinations "
                f"(max {MAX_GRID_COMBINATIONS}). Reduce your search space.",
            )
        return [dict(zip(keys, combo, strict=True)) for combo in combinations]

    def sample_random_combination(self, search_space: dict, rng: np.random.Generator) -> dict:
        """Sample one hyperparameter combination from the search space.

        Supports three spec types:
          - ``list``: uniform choice from the list.
          - ``{"type": "log_uniform", "min": ..., "max": ...}``: log-uniform.
          - ``{"type": "uniform", "min": ..., "max": ...}``: uniform.
        """
        params: dict = {}
        for key, spec in search_space.items():
            if isinstance(spec, list):
                chosen = rng.choice(spec)
                # Convert numpy types to native Python for JSON serialisation.
                params[key] = chosen.item() if hasattr(chosen, "item") else chosen
            elif isinstance(spec, dict) and spec.get("type") == "log_uniform":
                log_min = np.log(spec["min"])
                log_max = np.log(spec["max"])
                params[key] = float(np.exp(rng.uniform(log_min, log_max)))
            elif isinstance(spec, dict) and spec.get("type") == "uniform":
                params[key] = float(rng.uniform(spec["min"], spec["max"]))
        return params

    def validate_search_space(self, search_space: dict, strategy: str) -> None:
        """Validate a search space and raise ``AppException(400)`` if it is invalid.

        Guards against silent failures:
          - an empty search space or empty list values would otherwise produce a
            zero-trial session that completes without doing any work;
          - malformed dict specs (missing/invalid min/max) would otherwise crash
            the background tuning thread, leaving the session stuck in ``RUNNING``;
          - grid search only supports discrete (list) parameters, so a search space
            with no list parameter would yield a single useless ``{}`` trial.
        """
        if not isinstance(search_space, dict) or not search_space:
            raise AppException(400, "search_space must be a non-empty object")

        has_discrete = False

        for key, spec in search_space.items():
            if isinstance(spec, list):
                if not spec:
                    raise AppException(
                        400,
                        f"search_space['{key}'] must be a non-empty list",
                    )
                has_discrete = True
            elif isinstance(spec, dict) and spec.get("type") in ("uniform", "log_uniform"):
                try:
                    min_val = float(spec["min"])
                    max_val = float(spec["max"])
                except (KeyError, TypeError, ValueError):
                    raise AppException(
                        400,
                        f"search_space['{key}'] must define numeric 'min' and 'max'",
                    ) from None
                if min_val >= max_val:
                    raise AppException(
                        400,
                        f"search_space['{key}'] must have 'min' < 'max'",
                    )
                if spec.get("type") == "log_uniform" and min_val <= 0:
                    raise AppException(
                        400,
                        f"search_space['{key}'] must have 'min' > 0 for log_uniform",
                    )
            else:
                raise AppException(
                    400,
                    f"search_space['{key}'] must be a non-empty list or a 'uniform'/'log_uniform' spec dict",
                )

        if strategy == TuningStrategy.GRID and not has_discrete:
            raise AppException(
                400,
                "Grid search requires at least one discrete (list) parameter",
            )

    # ------------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------------

    def _run_single_trial(
        self,
        session_id: str,
        model_name: str,
        hyperparams: dict,
        trial_num: int,
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        """Create a training-job row, run training with a timeout, return job_id.

        Runs synchronously in the tuning background thread.
        """
        from app.services.model_run import model_run
        from app.services.training_service import create_training_job, update_job_status

        with make_session() as session:
            model = session.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
            if model is None:
                raise AppException(404, f"Model '{model_name}' not found")

            job = create_training_job(
                model_id=model.id,
                project_id=model.project_id,
                hyperparams=hyperparams,
                session=session,
            )
            job_id = job.id

            # Link job to tuning session.
            job.tuning_session_id = session_id
            session.add(job)
            session.commit()

        # Apply hyperparams to the model config temporarily for this run.
        # We modify the DB row, run training, then restore. But since model_run
        # reads the model config from the DB, we need to update it first.
        self._apply_hyperparams_to_model(model_name, hyperparams)

        timer = threading.Timer(TRIAL_TIMEOUT_SECONDS, self._timeout_trial, args=[job_id])
        timer.start()
        try:
            with make_session() as session:
                model_run(model_name, session, loop=loop, job_id=job_id)
        except Exception as e:
            logger.error("Trial %d failed for tuning session %s: %s", trial_num, session_id, e)
            with make_session() as session:
                update_job_status(job_id, TrainingStatus.FAILED, session, error_message=str(e))
        finally:
            timer.cancel()

        # Restore model config after trial.
        self._restore_model_config(model_name)

        return job_id

    def _apply_hyperparams_to_model(self, model_name: str, hyperparams: dict) -> None:
        """Temporarily apply hyperparams to the model's DB config for a trial run."""
        with make_session() as session:
            model = session.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
            if model is None:
                return
            # Store original values for restoration.
            if not hasattr(self, "_original_configs"):
                self._original_configs = {}
            self._original_configs[model_name] = {
                "optimizer": model.optimizer,
                "epochs": model.epochs,
                "batch_size": model.batch_size,
            }
            # Apply trial hyperparams.
            if "optimizer" in hyperparams:
                model.optimizer = hyperparams["optimizer"]
            if "epochs" in hyperparams:
                model.epochs = int(hyperparams["epochs"])
            if "batch_size" in hyperparams:
                model.batch_size = int(hyperparams["batch_size"])
            session.add(model)
            session.commit()

    def _restore_model_config(self, model_name: str) -> None:
        """Restore model config to pre-trial values."""
        originals = getattr(self, "_original_configs", {}).get(model_name)
        if originals is None:
            return
        with make_session() as session:
            model = session.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
            if model is None:
                return
            model.optimizer = originals["optimizer"]
            model.epochs = originals["epochs"]
            model.batch_size = originals["batch_size"]
            session.add(model)
            session.commit()

    def _timeout_trial(self, job_id: str) -> None:
        """Called by ``threading.Timer`` if a trial exceeds the timeout.

        Sets the job's DB status to CANCELLED so ``CancellationCheckCallback``
        stops the Keras loop at the next epoch boundary.
        """
        logger.warning("Trial timeout exceeded for job %s", job_id)
        from app.services.training_service import update_job_status

        with make_session() as session:
            update_job_status(job_id, TrainingStatus.CANCELLED, session, error_message="Trial timeout exceeded")

    def _get_trial_metric(self, job_id: str, metric_name: str) -> float | None:
        """Read the final (last epoch) value of ``metric_name`` for a job.

        Falls back to common metric name variants (e.g. accuracy vs val_accuracy).
        Returns None if no metrics are recorded.
        """
        with make_session() as session:
            # Try the exact metric name first.
            row = session.exec(
                select(TrainingMetric)
                .where(TrainingMetric.job_id == job_id, TrainingMetric.metric_name == metric_name)
                .order_by(TrainingMetric.epoch.desc())
                .limit(1)
            ).first()
            if row is not None:
                return row.metric_value

            # Fallback: try without the "val_" prefix or with it.
            alt_name = metric_name.replace("val_", "") if metric_name.startswith("val_") else f"val_{metric_name}"
            row = session.exec(
                select(TrainingMetric)
                .where(TrainingMetric.job_id == job_id, TrainingMetric.metric_name == alt_name)
                .order_by(TrainingMetric.epoch.desc())
                .limit(1)
            ).first()
            return row.metric_value if row is not None else None

    # ------------------------------------------------------------------
    # Main tuning loop
    # ------------------------------------------------------------------

    def _emit_progress(
        self,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        data: dict,
    ) -> None:
        """Thread-safely emit a tuning progress event to the session's room."""
        if loop is None or not loop.is_running():
            logger.warning("No running event loop for tuning emit (session %s)", session_id)
            return
        future = asyncio.run_coroutine_threadsafe(
            sio.emit(
                "tuning_progress",
                data,
                room=f"tuning:{session_id}",
                namespace=SOCKETIO_DL_NAMESPACE,
            ),
            loop,
        )
        try:
            future.result(timeout=5)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to emit tuning progress for session %s", session_id)

    def _fail_tuning_session(
        self, session_id: str, loop: asyncio.AbstractEventLoop, message: str | None = None
    ) -> None:
        """Mark a tuning session as FAILED and emit a completion event.

        Ensures a session can never be left stuck in ``RUNNING`` with no
        completion event when an unrecoverable error occurs.
        """
        with make_session() as session:
            ts = session.get(TuningSession, session_id)
            if ts is None:
                return
            if ts.status != TuningStatus.CANCELLED:
                ts.status = TuningStatus.FAILED
            ts.completed_at = _utcnow()
            session.add(ts)
            session.commit()
            final_status = ts.status.value
        self._emit_progress(
            loop,
            session_id,
            {
                "type": "tuning_complete",
                "status": final_status,
                "best_job_id": None,
                "best_metric": None,
                "error": message,
            },
        )

    def run_tuning_loop(
        self,
        session_id: str,
        model_name: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Main tuning loop.  Runs sequentially in a background thread.

        Generates trial combinations, executes each one, emits progress,
        and handles early stopping and cancellation.
        """
        # Load session.
        with make_session() as session:
            tuning_session = session.get(TuningSession, session_id)
            if tuning_session is None:
                logger.error("Tuning session %s not found", session_id)
                return
            tuning_session.status = TuningStatus.RUNNING
            session.add(tuning_session)
            session.commit()

            strategy = tuning_session.strategy
            search_space = tuning_session.search_space
            max_trials = tuning_session.max_trials
            metric = tuning_session.metric
            direction = tuning_session.direction
            early_stop_threshold = tuning_session.early_stop_threshold

        # Validate the search space defensively so a malformed row can never
        # crash the loop below and leave the session stuck in RUNNING.
        try:
            self.validate_search_space(search_space, strategy)
        except AppException as e:
            logger.error("Tuning session %s has an invalid search space: %s", session_id, e.detail)
            self._fail_tuning_session(session_id, loop, str(e.detail))
            return

        # Generate trial combinations.
        if strategy == TuningStrategy.GRID:
            trials = self.generate_grid_combinations(search_space)
        else:
            rng = np.random.default_rng(seed=42)
            trials = [self.sample_random_combination(search_space, rng) for _ in range(max_trials)]

        # Update total trials.
        with make_session() as session:
            ts = session.get(TuningSession, session_id)
            ts.total_trials = len(trials)
            session.add(ts)
            session.commit()

        best_metric: float | None = None
        best_job_id: str | None = None

        for trial_num, hyperparams in enumerate(trials):
            # Check for cancellation.
            with make_session() as session:
                ts = session.get(TuningSession, session_id)
                if ts.status == TuningStatus.CANCELLED:
                    logger.info("Tuning session %s was cancelled, stopping", session_id)
                    break

            # Run the trial.
            try:
                job_id = self._run_single_trial(session_id, model_name, hyperparams, trial_num, loop)
            except Exception as e:
                logger.error("Trial %d failed: %s", trial_num, e)
                # Update completed count even on failure.
                with make_session() as session:
                    ts = session.get(TuningSession, session_id)
                    ts.completed_trials += 1
                    session.add(ts)
                    session.commit()
                continue

            # Get the trial's target metric value.
            final_metric = self._get_trial_metric(job_id, metric)

            # Update best.
            if final_metric is not None and (
                best_metric is None
                or (direction == "maximize" and final_metric > best_metric)
                or (direction == "minimize" and final_metric < best_metric)
            ):
                best_metric = final_metric
                best_job_id = job_id

            # Update session progress.
            with make_session() as session:
                ts = session.get(TuningSession, session_id)
                ts.completed_trials += 1
                if best_job_id is not None:
                    ts.best_job_id = best_job_id
                session.add(ts)
                session.commit()

            # Emit progress.
            self._emit_progress(
                loop,
                session_id,
                {
                    "type": "tuning_progress",
                    "trial": trial_num + 1,
                    "total": len(trials),
                    "job_id": job_id,
                    "hyperparams": hyperparams,
                    "metric": final_metric,
                    "best_metric": best_metric,
                    "best_job_id": best_job_id,
                },
            )

            # Early stopping check.
            if early_stop_threshold is not None and final_metric is not None:
                should_stop = (direction == "maximize" and final_metric >= early_stop_threshold) or (
                    direction == "minimize" and final_metric <= early_stop_threshold
                )
                if should_stop:
                    logger.info(
                        "Early stopping triggered for session %s: metric=%s threshold=%s",
                        session_id,
                        final_metric,
                        early_stop_threshold,
                    )
                    with make_session() as session:
                        ts = session.get(TuningSession, session_id)
                        ts.early_stopped = True
                        session.add(ts)
                        session.commit()
                    break

        # Mark session complete (unless cancelled).
        with make_session() as session:
            ts = session.get(TuningSession, session_id)
            if ts.status != TuningStatus.CANCELLED:
                ts.status = TuningStatus.COMPLETED
            ts.completed_at = _utcnow()
            if best_job_id is not None:
                ts.best_job_id = best_job_id
            session.add(ts)
            session.commit()
            final_status = ts.status.value

        # Emit completion event.
        self._emit_progress(
            loop,
            session_id,
            {
                "type": "tuning_complete",
                "status": final_status,
                "best_job_id": best_job_id,
                "best_metric": best_metric,
            },
        )

    # ------------------------------------------------------------------
    # Duration estimation
    # ------------------------------------------------------------------

    def estimate_session_duration(self, model_id: int, n_trials: int) -> int:
        """Return estimated total duration in seconds.

        Uses the most recent completed job's wall-clock time for the same model.
        Falls back to ``DEFAULT_TRIAL_ESTIMATE_SECONDS`` if no history exists.
        """
        with make_session() as session:
            last_job = session.exec(
                select(TrainingJob)
                .where(TrainingJob.model_id == model_id, TrainingJob.status == TrainingStatus.COMPLETED)
                .order_by(TrainingJob.started_at.desc())
                .limit(1)
            ).first()
            if last_job and last_job.started_at and last_job.completed_at:
                per_trial = (last_job.completed_at - last_job.started_at).total_seconds()
            else:
                per_trial = DEFAULT_TRIAL_ESTIMATE_SECONDS
        return int(per_trial * n_trials)

    # ------------------------------------------------------------------
    # Apply best
    # ------------------------------------------------------------------

    def apply_best_params(self, session_id: str) -> dict:
        """Copy the best trial's hyperparams to the model's training config.

        Returns the applied hyperparams dict.
        Raises ``AppException`` if the session or best job is not found.
        """
        with make_session() as session:
            ts = session.get(TuningSession, session_id)
            if ts is None:
                raise AppException(404, "Tuning session not found")
            if ts.best_job_id is None:
                raise AppException(400, "No best trial found for this tuning session")

            best_job = session.get(TrainingJob, ts.best_job_id)
            if best_job is None or best_job.hyperparams is None:
                raise AppException(400, "Best trial has no hyperparams")

            model = session.get(ModelBasic, ts.model_id)
            if model is None:
                raise AppException(404, "Model not found")

            hp = best_job.hyperparams
            if "optimizer" in hp:
                model.optimizer = hp["optimizer"]
            if "epochs" in hp:
                model.epochs = int(hp["epochs"])
            if "batch_size" in hp:
                model.batch_size = int(hp["batch_size"])

            session.add(model)
            session.commit()
            return hp


# Module-level singleton.
tuning_service = TuningService()


def launch_tuning_session(session_id: str, model_name: str, loop: asyncio.AbstractEventLoop) -> None:
    """Schedule a tuning session to run in the background.

    Uses ``asyncio.to_thread`` wrapped in a tracked task so the event loop
    keeps a strong reference until it finishes.
    """
    task = asyncio.create_task(asyncio.to_thread(tuning_service.run_tuning_loop, session_id, model_name, loop))
    _tuning_tasks.add(task)
    task.add_done_callback(_tuning_tasks.discard)
