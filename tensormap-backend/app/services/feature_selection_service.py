import shutil
import uuid as uuid_pkg
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sqlmodel import Session, select

from app.config import get_settings
from app.exceptions import AppException
from app.models.data import DataFile, DataProcess
from app.models.ml import ModelBasic
from app.schemas.feature_selection import (
    ApplyFeatureSelectionRequest,
    ApplyFeatureSelectionResponse,
    FeatureMetricItem,
    FeatureSelectionAnalyzeRequest,
    FeatureSelectionAnalyzeResponse,
    RelevanceTier,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class FeatureSelectionService:
    """Computes non-linear and linear feature relevance metrics (Mutual Information,

    ANOVA F-statistics, sample variance) and executes automated feature pruning.
    """

    def _get_data_file_and_path(self, db: Session, file_id: uuid_pkg.UUID) -> tuple[DataFile, Path]:
        """Validates that the file exists in the database and disk and is a CSV."""
        file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
        if not file:
            raise AppException(404, f"Dataset file {file_id} not found in database")

        if file.file_type and file.file_type.lower() != "csv":
            raise AppException(400, "Feature selection is only supported for tabular CSV datasets")

        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / file.disk_name
        if not file_path.exists():
            raise AppException(404, f"Dataset file not found on disk at {file_path}")

        return file, file_path

    def _resolve_target_and_problem_type(
        self,
        db: Session,
        file_id: uuid_pkg.UUID,
        df: pd.DataFrame,
        target_override: str | None,
        problem_type_override: str | None,
    ) -> tuple[str, str]:
        """Resolves target column and problem type ('classification' vs 'regression')."""
        target = target_override

        if not target:
            dp = db.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first()
            if dp and dp.target:
                target = dp.target

        if not target:
            mb = db.exec(select(ModelBasic).where(ModelBasic.file_id == file_id)).first()
            if mb and mb.target_field:
                target = mb.target_field

        if not target:
            raise AppException(
                400,
                "No target field specified and no active target configured for this dataset. "
                "Please specify target_field in the request or set a target in Preprocessing.",
            )

        if target not in df.columns:
            raise AppException(
                400,
                f"Target column '{target}' not found in dataset columns: {list(df.columns)}",
            )

        # Resolve problem type
        if problem_type_override and problem_type_override.lower() in ("classification", "regression"):
            problem_type = problem_type_override.lower()
        else:
            target_series = df[target].dropna()
            if target_series.empty:
                raise AppException(400, f"Target column '{target}' contains only missing values")

            if not pd.api.types.is_numeric_dtype(target_series):
                problem_type = "classification"
            else:
                n_unique = target_series.nunique()
                n_total = len(target_series)
                # If numeric values are few discrete integers (<= 10 or <= 5% of dataset), infer classification
                if n_unique <= 10 or (n_total > 50 and n_unique / n_total <= 0.05):
                    problem_type = "classification"
                else:
                    problem_type = "regression"

        return target, problem_type

    def analyze_feature_relevance(
        self,
        db: Session,
        file_id: uuid_pkg.UUID,
        request: FeatureSelectionAnalyzeRequest,
    ) -> FeatureSelectionAnalyzeResponse:
        """Calculates Mutual Information, ANOVA F-statistic, variance, and composite

        relevance tiers for all candidate features against the target column.
        """
        _, file_path = self._get_data_file_and_path(db, file_id)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise AppException(400, f"Failed to parse CSV dataset: {e}") from e

        if df.empty:
            raise AppException(400, "Dataset is empty")

        target_col, problem_type = self._resolve_target_and_problem_type(
            db, file_id, df, request.target_field, request.problem_type
        )

        candidate_cols = [c for c in df.columns if c != target_col]
        if not candidate_cols:
            raise AppException(400, "No candidate predictive features found in dataset")

        # Clean dataset: drop rows where target is missing
        clean_df = df.dropna(subset=[target_col]).copy()
        if clean_df.empty:
            raise AppException(400, f"Target column '{target_col}' has no valid non-null rows")

        # Encode target
        y_raw = clean_df[target_col]
        if problem_type == "classification":
            y_encoded = pd.Categorical(y_raw).codes
            if len(np.unique(y_encoded)) < 2:
                raise AppException(
                    400,
                    f"Classification target '{target_col}' must contain at least 2 distinct classes. "
                    f"Found only {len(np.unique(y_encoded))} class.",
                )
        else:
            try:
                y_encoded = pd.to_numeric(y_raw, errors="coerce")
                if y_encoded.isna().all():
                    raise AppException(400, f"Regression target '{target_col}' contains no valid numeric values")
                # Drop any rows where numeric coercion produced NaN
                valid_mask = ~y_encoded.isna()
                clean_df = clean_df[valid_mask]
                y_encoded = y_encoded[valid_mask].values.astype(float)
            except Exception as e:
                raise AppException(400, f"Failed to parse regression target '{target_col}' as numeric: {e}") from e

        total_samples = len(clean_df)
        if total_samples < 4:
            raise AppException(
                400,
                f"Dataset has only {total_samples} valid rows; at least 4 rows are required for statistical analysis.",
            )

        # Prepare feature matrix X
        X_df = clean_df[candidate_cols].copy()
        feature_matrices: list[np.ndarray] = []
        variances: list[float] = []
        is_constants: list[bool] = []

        for col in candidate_cols:
            s = X_df[col]
            if pd.api.types.is_numeric_dtype(s):
                num_s = pd.to_numeric(s, errors="coerce")
                median_val = num_s.median()
                fill_val = median_val if pd.notna(median_val) else 0.0
                num_s = num_s.fillna(fill_val)
                var = float(np.var(num_s.values, ddof=1)) if len(num_s) > 1 else 0.0
                is_const = bool(var < 1e-9 or np.isclose(num_s.min(), num_s.max()))
                variances.append(max(0.0, round(var, 6)))
                is_constants.append(is_const)
                feature_matrices.append(num_s.values.astype(float))
            else:
                # Categorical column: encode with Categorical.codes
                cat_codes = pd.Categorical(s.fillna("_missing_")).codes.astype(float)
                var = float(np.var(cat_codes, ddof=1)) if len(cat_codes) > 1 else 0.0
                is_const = bool(var < 1e-9 or len(np.unique(cat_codes)) <= 1)
                variances.append(max(0.0, round(var, 6)))
                is_constants.append(is_const)
                feature_matrices.append(cat_codes)

        X_array = np.column_stack(feature_matrices)

        # 1. Mutual Information Estimation
        n_neighbors = min(request.n_neighbors, max(1, total_samples - 1))
        try:
            if problem_type == "classification":
                mi_scores = mutual_info_classif(
                    X_array,
                    y_encoded,
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
            else:
                mi_scores = mutual_info_regression(
                    X_array,
                    y_encoded,
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
        except Exception as e:
            logger.warning("Mutual information estimation fallback: %s", e)
            mi_scores = np.zeros(len(candidate_cols))

        # 2. Parametric F-statistic & p-value
        f_scores = np.zeros(len(candidate_cols))
        p_values = np.ones(len(candidate_cols))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if problem_type == "classification":
                    f_res, p_res = f_classif(X_array, y_encoded)
                else:
                    f_res, p_res = f_regression(X_array, y_encoded)

                f_scores = np.nan_to_num(f_res, nan=0.0, posinf=0.0, neginf=0.0)
                p_values = np.nan_to_num(p_res, nan=1.0, posinf=1.0, neginf=1.0)
            except Exception as e:
                logger.warning("F-statistic calculation fallback: %s", e)

        # 3. Normalized Composite Importance Score (0.0 to 100.0)
        max_mi = float(np.max(mi_scores)) if len(mi_scores) > 0 and np.max(mi_scores) > 0 else 1.0
        max_f = float(np.max(f_scores)) if len(f_scores) > 0 and np.max(f_scores) > 0 else 1.0

        items: list[FeatureMetricItem] = []
        critical_count = 0
        informative_count = 0
        weak_count = 0
        constant_count = 0

        for i, col in enumerate(candidate_cols):
            mi_val = max(0.0, float(mi_scores[i]))
            f_val = max(0.0, float(f_scores[i]))
            p_val = float(p_values[i])
            var_val = variances[i]
            is_const = is_constants[i]

            if is_const:
                norm_score = 0.0
                tier = RelevanceTier.CONSTANT_OR_REDUNDANT
                action = "prune"
                constant_count += 1
            else:
                mi_part = mi_val / max_mi if max_mi > 0 else 0.0
                f_part = f_val / max_f if max_f > 0 else 0.0
                # Weighted composite score: 60% non-linear MI, 40% linear F-statistic
                norm_score = round(float(0.60 * mi_part + 0.40 * f_part) * 100.0, 2)

                if norm_score >= 50.0:
                    tier = RelevanceTier.CRITICAL
                    action = "keep"
                    critical_count += 1
                elif norm_score >= 20.0:
                    tier = RelevanceTier.INFORMATIVE
                    action = "keep"
                    informative_count += 1
                elif norm_score >= 5.0:
                    tier = RelevanceTier.WEAK
                    action = "review"
                    weak_count += 1
                else:
                    tier = RelevanceTier.CONSTANT_OR_REDUNDANT
                    action = "prune"
                    constant_count += 1

            items.append(
                FeatureMetricItem(
                    feature=col,
                    mutual_info=round(mi_val, 4),
                    f_statistic=round(f_val, 4) if f_val > 0 else None,
                    p_value=round(p_val, 6) if p_val < 1.0 else None,
                    variance=round(var_val, 6),
                    normalized_score=norm_score,
                    tier=tier,
                    is_constant=is_const,
                    suggested_action=action,
                )
            )

        # Sort features by normalized score descending
        items.sort(key=lambda item: item.normalized_score, reverse=True)

        return FeatureSelectionAnalyzeResponse(
            file_id=str(file_id),
            target_field=target_col,
            problem_type=problem_type,
            total_samples=total_samples,
            total_features=len(candidate_cols),
            critical_features_count=critical_count,
            informative_features_count=informative_count,
            weak_features_count=weak_count,
            constant_features_count=constant_count,
            features=items,
        )

    def apply_feature_selection(
        self,
        db: Session,
        file_id: uuid_pkg.UUID,
        request: ApplyFeatureSelectionRequest,
    ) -> ApplyFeatureSelectionResponse:
        """Prunes the dataset CSV down to selected features plus target, updating

        the database record and creating a backup copy.
        """
        data_file, file_path = self._get_data_file_and_path(db, file_id)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise AppException(400, f"Failed to parse CSV dataset: {e}") from e

        target_col, _ = self._resolve_target_and_problem_type(
            db, file_id, df, target_override=None, problem_type_override=None
        )

        all_candidate_features = [c for c in df.columns if c != target_col]

        # Determine which features to keep
        if request.selected_features is not None:
            # Explicit selection provided
            selected = list(dict.fromkeys(request.selected_features))  # Deduplicate preserving order
            unknown = [f for f in selected if f not in all_candidate_features]
            if unknown:
                raise AppException(
                    400,
                    f"Unknown feature(s) specified in selection: {unknown}. "
                    f"Available features: {all_candidate_features}",
                )
            chosen_features = [f for f in selected if f in all_candidate_features]
        elif request.top_k is not None:
            if request.top_k < 1:
                raise AppException(400, "top_k must be at least 1")
            # Analyze to rank features
            analysis = self.analyze_feature_relevance(
                db,
                file_id,
                FeatureSelectionAnalyzeRequest(target_field=target_col),
            )
            ranked_names = [item.feature for item in analysis.features]
            chosen_features = ranked_names[: request.top_k]
        elif request.min_score is not None:
            analysis = self.analyze_feature_relevance(
                db,
                file_id,
                FeatureSelectionAnalyzeRequest(target_field=target_col),
            )
            chosen_features = [item.feature for item in analysis.features if item.normalized_score >= request.min_score]
        else:
            raise AppException(400, "Must specify 'selected_features', 'top_k', or 'min_score' to prune dataset")

        if not chosen_features:
            raise AppException(
                400,
                "Cannot prune all features. At least 1 candidate predictive feature must be retained.",
            )

        dropped_features = [c for c in all_candidate_features if c not in chosen_features]

        # Final column order: chosen features followed by target
        final_cols = [c for c in chosen_features if c in df.columns]
        if target_col in df.columns and target_col not in final_cols:
            final_cols.append(target_col)

        pruned_df = df[final_cols]

        # Create backup copy if not already existing
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        backup_created = False
        try:
            shutil.copy2(file_path, backup_path)
            backup_created = True
        except OSError as e:
            logger.warning("Could not create backup file at %s: %s", backup_path, e)

        # Overwrite CSV with pruned DataFrame
        try:
            pruned_df.to_csv(file_path, index=False)
        except Exception as e:
            raise AppException(500, f"Failed to save pruned dataset to disk: {e}") from e

        # Update DataFile record in DB
        data_file.columns = final_cols
        data_file.row_count = len(pruned_df)
        db.add(data_file)
        db.commit()
        db.refresh(data_file)

        msg = (
            f"Successfully pruned dataset: retained {len(chosen_features)} features, "
            f"dropped {len(dropped_features)} features."
        )

        return ApplyFeatureSelectionResponse(
            file_id=str(file_id),
            target_field=target_col,
            retained_features=chosen_features,
            dropped_features=dropped_features,
            total_columns_remaining=len(final_cols),
            backup_created=backup_created,
            message=msg,
        )


feature_selection_service = FeatureSelectionService()
