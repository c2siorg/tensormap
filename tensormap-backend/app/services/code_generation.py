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
    MODEL_LOSS,
    MODEL_LOSS_EXPRESSION,
    MODEL_METRIC,
    MODEL_OPTIMIZER,
    MODEL_TRAINING_SPLIT,
    TEMPLATE_ROOT,
)
from app.shared.enums import LossFunction, ProblemType, default_loss_for_problem_type
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class CodeGenerationError(Exception):
    """Raised when code generation cannot proceed due to missing data."""

    pass


def _loss_expression(loss: str) -> str:
    """Return the Python expression string for compiling with the specified loss."""
    if loss == LossFunction.CATEGORICAL_CROSSENTROPY:
        return "tf.keras.losses.CategoricalCrossentropy(from_logits=True)"
    if loss == LossFunction.BINARY_CROSSENTROPY:
        return "tf.keras.losses.BinaryCrossentropy(from_logits=True)"
    if loss == LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY:
        return "tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)"
    return f'"{loss}"'


def generate_code(model_name: str, db: Session) -> str:
    """Generate TensorFlow Python code from a saved model configuration."""
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if model_configs is None:
        raise CodeGenerationError(f"Model not found: {model_name}")
    file = db.exec(select(DataFile).where(DataFile.id == model_configs.file_id)).first()
    if file is None:
        raise CodeGenerationError(f"Dataset file not found (id={model_configs.file_id})")

    loss = model_configs.loss or default_loss_for_problem_type(model_configs.model_type)

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
            MODEL_LOSS: loss,
            MODEL_LOSS_EXPRESSION: _loss_expression(loss),
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


def _map_template(problem_type_id: int | None) -> str:
    """Map a ProblemType enum value to its Jinja2 template path."""
    options = {
        ProblemType.CLASSIFICATION: CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py",
        ProblemType.REGRESSION: CODE_TEMPLATE_FOLDER + "linear-regression-all-float.py",
        ProblemType.IMAGE_CLASSIFICATION: CODE_TEMPLATE_FOLDER + "simple-image-classification.py",
    }
    if problem_type_id is not None and problem_type_id in options:
        return options[problem_type_id]
    valid = ", ".join(str(int(p)) for p in ProblemType)
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
    return file.file_name
