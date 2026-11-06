from enum import StrEnum

from pydantic import BaseModel, Field


class RelevanceTier(StrEnum):
    """Relevance and predictive value tier of a candidate feature."""

    CRITICAL = "CRITICAL"
    INFORMATIVE = "INFORMATIVE"
    WEAK = "WEAK"
    CONSTANT_OR_REDUNDANT = "CONSTANT_OR_REDUNDANT"


class FeatureMetricItem(BaseModel):
    """Detailed statistical and information-theoretic metrics for a single feature."""

    feature: str = Field(description="Name of the candidate feature column")
    mutual_info: float = Field(description="Estimated mutual information between feature and target (>= 0.0)")
    f_statistic: float | None = Field(
        default=None,
        description="ANOVA F-statistic (classification) or F-regression statistic (regression)",
    )
    p_value: float | None = Field(
        default=None,
        description="Associated p-value for the F-statistic test (null hypothesis of no association)",
    )
    variance: float = Field(description="Sample variance of feature values")
    normalized_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Normalized composite importance score scaled from 0.0 to 100.0",
    )
    tier: RelevanceTier = Field(description="Categorical relevance tier based on composite score and variance")
    is_constant: bool = Field(default=False, description="True if feature has zero or near-zero variance")
    suggested_action: str = Field(description="Automated recommendation for this feature: 'keep', 'review', or 'prune'")


class FeatureSelectionAnalyzeRequest(BaseModel):
    """Request payload to analyze feature relevance for a dataset."""

    target_field: str | None = Field(
        default=None,
        description="Target column name. If omitted, uses active target from DataProcess.",
    )
    problem_type: str | None = Field(
        default=None,
        description="Optional problem type override ('classification' or 'regression'). Auto-inferred if omitted.",
    )
    n_neighbors: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of neighbors for k-NN mutual information estimation",
    )


class FeatureSelectionAnalyzeResponse(BaseModel):
    """Complete feature relevance diagnostic report."""

    file_id: str = Field(description="Dataset file UUID")
    target_field: str = Field(description="Target column evaluated against")
    problem_type: str = Field(description="Resolved problem type ('classification' or 'regression')")
    total_samples: int = Field(description="Total valid non-null rows analyzed")
    total_features: int = Field(description="Total candidate predictive features analyzed")
    critical_features_count: int = Field(description="Count of features in CRITICAL tier")
    informative_features_count: int = Field(description="Count of features in INFORMATIVE tier")
    weak_features_count: int = Field(description="Count of features in WEAK tier")
    constant_features_count: int = Field(description="Count of features that are constant or zero-variance")
    features: list[FeatureMetricItem] = Field(description="Features ranked by normalized composite importance")


class ApplyFeatureSelectionRequest(BaseModel):
    """Request payload to prune dataset down to selected features."""

    selected_features: list[str] | None = Field(
        default=None,
        description="Explicit list of feature column names to retain in the dataset",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Automatically retain the top K highest scoring features",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Automatically retain all features with normalized composite score >= min_score",
    )


class ApplyFeatureSelectionResponse(BaseModel):
    """Result of pruning dataset down to selected features."""

    file_id: str = Field(description="Dataset file UUID")
    target_field: str = Field(description="Target column preserved in the dataset")
    retained_features: list[str] = Field(description="List of feature columns retained in updated dataset")
    dropped_features: list[str] = Field(description="List of feature columns dropped from dataset")
    total_columns_remaining: int = Field(description="Total column count in modified CSV (features + target)")
    backup_created: bool = Field(description="Whether a .bak backup file was created or preserved")
    message: str = Field(description="Human-readable result summary")
