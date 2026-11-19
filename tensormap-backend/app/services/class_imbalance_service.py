"""Service for diagnosing dataset class imbalance and calculating cost-sensitive class weights."""

import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.config import get_settings
from app.exceptions import AppException
from app.models.data import DataFile, DataProcess
from app.schemas.class_imbalance import (
    ClassDistributionItem,
    ClassImbalanceResponse,
    ImbalanceSeverity,
)


class ClassImbalanceService:
    """Computes class frequency metrics, imbalance ratios, Shannon entropy, and balanced weights."""

    def analyze_class_imbalance(
        self, db: Session, file_id: UUID, target_field: str | None = None
    ) -> ClassImbalanceResponse:
        """Inspects target distribution in a CSV dataset and computes imbalance diagnostics."""
        data_file = db.get(DataFile, file_id)
        if not data_file:
            raise AppException(404, f"Dataset file {file_id} not found")

        file_path = Path(get_settings().upload_folder) / data_file.disk_name
        if not file_path.exists():
            raise AppException(404, f"Dataset file not found on disk at {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise AppException(400, f"Failed to parse CSV dataset: {e}") from e

        resolved_target = target_field
        if not resolved_target:
            dp = db.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first()
            if dp and dp.target:
                resolved_target = dp.target

        if not resolved_target:
            raise AppException(400, "Target column must be provided via query param or preprocessing settings.")

        if resolved_target not in df.columns:
            raise AppException(
                400, f"Target column '{resolved_target}' not found in dataset columns: {list(df.columns)}"
            )

        target_series = df[resolved_target].dropna()
        if target_series.empty:
            raise AppException(400, f"Target column '{resolved_target}' contains only missing values")

        n_unique = target_series.nunique()
        if (
            pd.api.types.is_float_dtype(target_series)
            or (pd.api.types.is_numeric_dtype(target_series) and n_unique > 50)
            or n_unique > 100
        ):
            raise AppException(
                400,
                f"Class imbalance analysis requires a discrete classification target, but '{resolved_target}' "
                f"appears to be a continuous regression variable ({n_unique} unique values).",
            )

        counts_series = target_series.value_counts()
        try:
            counts_series = counts_series.sort_index()
        except TypeError:
            counts_series = counts_series.sort_index(key=lambda s: s.astype(str))
        n_classes = len(counts_series)
        if n_classes < 2:
            raise AppException(400, f"Classification target must have at least 2 distinct classes. Found: {n_classes}")

        total_samples = int(counts_series.sum())
        imbalance_ratio = round(int(counts_series.max()) / max(int(counts_series.min()), 1), 4)

        probs = counts_series.values / total_samples
        entropy = -float(np.sum([p * math.log2(p) for p in probs if p > 0]))
        normalized_entropy = round(min(1.0, max(0.0, entropy / (math.log2(n_classes) if n_classes > 1 else 1.0))), 4)
        gini_impurity = round(float(1.0 - np.sum(probs**2)), 4)

        if imbalance_ratio <= 1.5:
            severity, is_imbalanced = ImbalanceSeverity.BALANCED, False
            strategy = "Class distribution is well-balanced. Standard categorical cross-entropy loss is recommended."
        elif imbalance_ratio <= 3.0:
            severity, is_imbalanced = ImbalanceSeverity.MILD, True
            strategy = "Mild class imbalance observed. Monitor minority class recall and PR-AUC."
        elif imbalance_ratio <= 10.0:
            severity, is_imbalanced = ImbalanceSeverity.MODERATE, True
            strategy = "Moderate class imbalance. Apply balanced class weights or evaluate using macro-averaged F1."
        else:
            severity, is_imbalanced = ImbalanceSeverity.SEVERE, True
            strategy = (
                "Severe class imbalance. High risk of majority collapse. Use class weights or SMOTE oversampling."
            )

        distribution = [
            ClassDistributionItem(
                class_name=str(val),
                count=int(cnt),
                percentage=round((int(cnt) / total_samples) * 100.0, 2),
                balanced_weight=round(total_samples / (n_classes * max(int(cnt), 1)), 4),
            )
            for val, cnt in counts_series.items()
        ]

        return ClassImbalanceResponse(
            file_id=file_id,
            target_field=resolved_target,
            total_samples=total_samples,
            n_classes=n_classes,
            imbalance_ratio=imbalance_ratio,
            shannon_entropy=round(entropy, 4),
            normalized_entropy=normalized_entropy,
            gini_impurity=gini_impurity,
            severity=severity,
            is_imbalanced=is_imbalanced,
            recommended_strategy=strategy,
            class_distribution=distribution,
        )


class_imbalance_service = ClassImbalanceService()
