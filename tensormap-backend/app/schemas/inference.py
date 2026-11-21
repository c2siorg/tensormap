"""Request and response schemas for the interactive model inference playground."""

from pydantic import BaseModel, Field


class FeatureSchemaItem(BaseModel):
    """Metadata for a single input feature to drive dynamic frontend sliders."""

    name: str = Field(description="Name of the feature column")
    dtype: str = Field(default="float", description="Data type: float, int, or categorical")
    min: float | None = Field(default=None, description="Minimum observed value in the dataset")
    max: float | None = Field(default=None, description="Maximum observed value in the dataset")
    mean: float | None = Field(default=None, description="Mean value across dataset")
    median: float | None = Field(default=None, description="Median value across dataset")
    step: float | None = Field(default=0.1, description="Recommended step increment for UI sliders")
    default_value: float = Field(default=0.0, description="Recommended default input value (median)")


class SampleTestRecord(BaseModel):
    """A real sample from the test split for instant testing and ground truth comparison."""

    index: int = Field(description="Sample index in the test dataset")
    features: dict[str, float] = Field(description="Mapping of feature names to values")
    actual_class: int | None = Field(default=None, description="Actual ground truth class integer (if classification)")
    actual_class_name: str | None = Field(
        default=None, description="Actual ground truth class name (if classification)"
    )
    actual_value: float | None = Field(default=None, description="Actual ground truth target value (if regression)")


class ModelFeatureSchemaResponse(BaseModel):
    """Complete schema for a model's input features, target, and sample test rows."""

    job_id: str
    target_field: str
    problem_type: str  # "classification" or "regression"
    features: list[FeatureSchemaItem]
    class_names: list[str] | None = None
    sample_records: list[SampleTestRecord] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    """Payload to execute inference on custom or sample inputs."""

    inputs: dict[str, float] = Field(description="Dictionary mapping feature names to numeric values")


class ClassProbability(BaseModel):
    """Predicted probability for a specific target class."""

    class_index: int
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    """Inference result including prediction, confidence, probability distribution, and latency."""

    success: bool = True
    problem_type: str  # "classification" or "regression"
    predicted_class: int | None = None
    predicted_label: str | None = None
    confidence: float | None = None
    predicted_value: float | None = None
    probabilities: list[ClassProbability] | None = None
    latency_ms: float = Field(description="Inference execution duration in milliseconds")
    input_features: dict[str, float]
