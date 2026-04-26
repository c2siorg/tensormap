"""GAP-3: Compute detailed evaluation metrics after model training."""

import numpy as np

from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def compute_detailed_metrics(
    y_true,
    y_pred_raw,
    problem_type: int,
    test_loss: float | None = None,
    test_metric: float | None = None,
) -> dict:
    """Compute comprehensive evaluation metrics based on problem type."""
    metrics: dict = {
        "test_loss": test_loss,
        "test_metric": test_metric,
    }

    # Validate inputs early
    if y_true is None or y_pred_raw is None:
        logger.warning("Cannot compute metrics: missing y_true or y_pred_raw")
        return metrics

    try:
        if problem_type == ProblemType.REGRESSION:
            _add_regression_metrics(metrics, y_true, y_pred_raw)
        elif problem_type in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
            _add_classification_metrics(metrics, y_true, y_pred_raw)
    except Exception as exc:
        logger.warning("Could not compute detailed metrics: %s", exc)

    return metrics


def _add_regression_metrics(metrics: dict, y_true, y_pred_raw) -> None:
    """Add MAE, RMSE, and R² for regression problems."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = np.array(y_pred_raw).flatten()
    y_true = np.array(y_true).flatten()

    metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 6)
    metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6)
    metrics["r_squared"] = round(float(r2_score(y_true, y_pred)), 6)


def _add_classification_metrics(metrics: dict, y_true, y_pred_raw) -> None:
    """Add per-class metrics, confusion matrix, and ROC/AUC for classification."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred_raw = np.array(y_pred_raw)

    # Convert probabilities to class labels
    if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] > 1:
        y_pred = np.argmax(y_pred_raw, axis=1)
        y_pred_proba = y_pred_raw
    else:
        y_pred_proba = y_pred_raw.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

    y_true = np.array(y_true).flatten().astype(int)

    # Per-class precision, recall, F1
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for k, v in report.items():
        if k not in ("accuracy", "macro avg", "weighted avg"):
            per_class[str(k)] = {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1_score": round(v["f1-score"], 4),
                "support": int(v["support"]),
            }
    metrics["per_class_metrics"] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ROC AUC (binary only)
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        try:
            from sklearn.metrics import auc, roc_curve

            proba = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
            fpr, tpr, thresholds = roc_curve(y_true, proba)
            roc_auc = auc(fpr, tpr)
            metrics["roc_auc"] = round(float(roc_auc), 6)
            metrics["roc_curve_data"] = {
                "fpr": [round(float(x), 6) for x in fpr],
                "tpr": [round(float(x), 6) for x in tpr],
                "thresholds": [round(float(x), 6) for x in thresholds],
            }
        except Exception as exc:
            logger.warning("ROC AUC computation failed: %s", exc)
