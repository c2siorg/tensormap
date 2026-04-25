from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.database import get_db
from app.models.ml import ModelBasic
from app.models.training_run import ModelTrainingRun

router = APIRouter(tags=["Training Runs"])


@router.get("/model/{model_id}/training-runs")
def get_training_runs(model_id: int, db: Session = Depends(get_db)):
    """List all training runs for a model, newest first."""
    model = db.get(ModelBasic, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    runs = db.exec(
        select(ModelTrainingRun)
        .where(ModelTrainingRun.model_id == model_id)
        .order_by(ModelTrainingRun.created_on.desc())
    ).all()

    return {
        "model_id": model_id,
        "model_name": model.model_name,
        "total_runs": len(runs),
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "duration_seconds": r.duration_seconds,
                "epochs_configured": r.epochs_configured,
                "batch_size_configured": r.batch_size_configured,
                "final_train_loss": r.final_train_loss,
                "final_train_metric": r.final_train_metric,
                "final_val_loss": r.final_val_loss,
                "final_val_metric": r.final_val_metric,
                "metric_name": r.metric_name,
                "error_message": r.error_message,
            }
            for r in runs
        ],
    }


@router.get("/model/{model_id}/training-run/{run_id}/metrics")
def get_training_run_metrics(model_id: int, run_id: int, db: Session = Depends(get_db)):
    """Get detailed epoch-by-epoch metrics for a specific training run."""
    run = db.exec(
        select(ModelTrainingRun).where(
            ModelTrainingRun.id == run_id,
            ModelTrainingRun.model_id == model_id,
        )
    ).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return {
        "id": run.id,
        "model_id": run.model_id,
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "duration_seconds": run.duration_seconds,
        "config": {
            "epochs": run.epochs_configured,
            "batch_size": run.batch_size_configured,
            "training_split": run.training_split_configured,
            "optimizer": run.optimizer,
            "loss_fn": run.loss_fn,
            "metric": run.metric_name,
        },
        "results": {
            "final_train_loss": run.final_train_loss,
            "final_train_metric": run.final_train_metric,
            "final_val_loss": run.final_val_loss,
            "final_val_metric": run.final_val_metric,
        },
        "curves": {
            "epoch_losses": run.epoch_losses,
            "epoch_metrics": run.epoch_metrics,
            "epoch_val_losses": run.epoch_val_losses,
            "epoch_val_metrics": run.epoch_val_metrics,
        },
        "error_message": run.error_message,
    }


@router.post("/model/{model_id}/training-run/{run_id}/set-as-best")
def set_as_best(model_id: int, run_id: int, db: Session = Depends(get_db)):
    """Mark a training run as the best run for this model."""
    run = db.exec(
        select(ModelTrainingRun).where(
            ModelTrainingRun.id == run_id,
            ModelTrainingRun.model_id == model_id,
        )
    ).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status not in ("success", "best"):
        raise HTTPException(status_code=400, detail="Can only mark successful runs as best")

    # Clear previous best for this model
    all_runs = db.exec(select(ModelTrainingRun).where(ModelTrainingRun.model_id == model_id)).all()
    for r in all_runs:
        if r.status == "best":
            r.status = "success"
            db.add(r)

    run.status = "best"
    db.add(run)
    db.commit()
    db.refresh(run)

    return {"message": f"Run #{run_id} marked as best", "run_id": run_id}
