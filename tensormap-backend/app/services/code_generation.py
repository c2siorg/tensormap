import contextlib

from jinja2 import Environment, FileSystemLoader
from sqlmodel import Session, select

from app.models import DataFile, ImageProperties, ModelBasic
from app.shared.constants import (
    BATCH_SIZE,
    CODE_TEMPLATE_FOLDER,
    COLOR_MODE,
    DATASET,
    FILE_NAME,
    FILE_TARGET,
    IMG_SIZE,
    LABEL_MODE,
    LEARNING_MODEL,
    MODEL_EPOCHS,
    MODEL_GENERATION_TYPE,
    MODEL_JSON_FILE,
    MODEL_METRIC,
    MODEL_OPTIMIZER,
    MODEL_TRAINING_SPLIT,
    TEMPLATE_ROOT,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class CodeGenerationError(Exception):
    """Raised when code generation cannot proceed due to missing data."""

    pass


def generate_code(model_name: str, db: Session) -> str:
    """Generate TensorFlow Python code from a saved model configuration."""
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if model_configs is None:
        raise CodeGenerationError(f"Model not found: {model_name}")
    file = db.exec(select(DataFile).where(DataFile.id == model_configs.file_id)).first()
    if file is None:
        raise CodeGenerationError(f"Dataset file not found (id={model_configs.file_id})")

    data = {
        DATASET: {
            FILE_NAME: _file_location(file),
            FILE_TARGET: model_configs.target_field,
            MODEL_TRAINING_SPLIT: model_configs.training_split,
        },
        LEARNING_MODEL: {
            MODEL_JSON_FILE: model_name + MODEL_GENERATION_TYPE,
            MODEL_OPTIMIZER: model_configs.optimizer,
            MODEL_METRIC: model_configs.metric,
            MODEL_EPOCHS: model_configs.epochs,
        },
    }

    if file.file_type == "zip":
        image_prop = db.exec(select(ImageProperties).where(ImageProperties.id == model_configs.file_id)).first()
        if image_prop:
            data[DATASET].update(
                {
                    IMG_SIZE: image_prop.image_size,
                    BATCH_SIZE: image_prop.batch_size,
                    COLOR_MODE: image_prop.color_mode,
                    LABEL_MODE: image_prop.label_mode,
                }
            )

    logger.debug("Generating code for model type: %s", model_configs.model_type)
    template_loader = FileSystemLoader(searchpath=TEMPLATE_ROOT)
    template_env = Environment(loader=template_loader)
    template = template_env.get_template(_map_template(model_configs.model_type))

    return template.render(data=data)


def _map_template(problem_type_id: int | str | ProblemType | None) -> str:
    """Map a ProblemType enum value to its Jinja2 template path."""
    options = {
        ProblemType.CLASSIFICATION: CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py",
        ProblemType.REGRESSION: CODE_TEMPLATE_FOLDER + "linear-regression-all-float.py",
        ProblemType.IMAGE_CLASSIFICATION: CODE_TEMPLATE_FOLDER + "simple-image-classification.py",
    }
    resolved_type: ProblemType | None = None
    if isinstance(problem_type_id, ProblemType):
        resolved_type = problem_type_id
    elif isinstance(problem_type_id, int):
        with contextlib.suppress(ValueError):
            resolved_type = ProblemType(problem_type_id)
    elif isinstance(problem_type_id, str):
        cleaned = problem_type_id.strip()
        if cleaned.isdigit():
            with contextlib.suppress(ValueError):
                resolved_type = ProblemType(int(cleaned))
        else:
            with contextlib.suppress(KeyError):
                resolved_type = ProblemType[cleaned.upper()]

    if resolved_type in options:
        return options[resolved_type]

    valid = ", ".join(f"{p.name} ({int(p)})" for p in ProblemType)
    raise ValueError(f"Unknown model type: {problem_type_id!r}. Valid: {valid}")


def _file_location(file: DataFile) -> str:
    """Return the dataset path used in the generated training script.

    Returns the original uploaded filename (or the ZIP extraction
    directory name) so the script references a path the user recognises.
    The user places the dataset beside the script; the server-local
    storage path is not embedded in the generated code.
    """
    if file.file_type == "zip":
        return file.file_name.rsplit(".", 1)[0]
    if file.file_type == "csv" and not file.file_name.lower().endswith(".csv"):
        return f"{file.file_name}.csv"
    return file.file_name
