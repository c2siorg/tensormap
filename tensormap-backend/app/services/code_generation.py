import json
import os

from jinja2 import Environment, FileSystemLoader
from sqlmodel import Session, select

from app.models import DataFile, ImageProperties, ModelBasic
from app.services.model_run import _emits_logits
from app.shared.constants import (
    BATCH_SIZE,
    CLASS_WEIGHT_MODE,
    CODE_TEMPLATE_FOLDER,
    COLOR_MODE,
    DATASET,
    FILE_NAME,
    FILE_TARGET,
    IMG_SIZE,
    LABEL_MODE,
    LEARNING_MODEL,
    MODEL_EPOCHS,
    MODEL_GENERATION_LOCATION,
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


def _model_emits_logits(model_name: str, model_configs: ModelBasic) -> bool:
    """Determine whether the model's output layer emits logits (True) or probabilities (False)."""
    json_path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    json_str: str | None = None
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                json_str = f.read()
        except OSError as e:
            logger.warning("Failed to read model JSON file for %r at %s: %s", model_name, json_path, e)

    if not json_str and model_configs.graph_json:
        if isinstance(model_configs.graph_json, str):
            json_str = model_configs.graph_json
        elif isinstance(model_configs.graph_json, dict):
            try:
                json_str = json.dumps(model_configs.graph_json)
            except (TypeError, ValueError) as e:
                logger.warning("Failed to serialize graph_json for model %r: %s", model_name, e)

    if json_str:
        try:
            return _emits_logits(json_str)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to parse model JSON or detect logits for %r: %s", model_name, e)

    return True


def _loss_expression(loss: str, from_logits: bool = True) -> str:
    """Return the Python expression string for compiling with the specified loss."""
    from_logits_str = "True" if from_logits else "False"
    if loss == LossFunction.CATEGORICAL_CROSSENTROPY:
        return f"tf.keras.losses.CategoricalCrossentropy(from_logits={from_logits_str})"
    if loss == LossFunction.BINARY_CROSSENTROPY:
        return f"tf.keras.losses.BinaryCrossentropy(from_logits={from_logits_str})"
    if loss == LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY:
        return f"tf.keras.losses.SparseCategoricalCrossentropy(from_logits={from_logits_str})"
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
    from_logits = _model_emits_logits(model_name, model_configs)

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
            MODEL_LOSS_EXPRESSION: _loss_expression(loss, from_logits=from_logits),
            BATCH_SIZE: model_configs.batch_size or 32,
            CLASS_WEIGHT_MODE: getattr(model_configs, "class_weight_mode", None),
        },
    }

    if file.file_type == "zip":
        image_prop = db.exec(select(ImageProperties).where(ImageProperties.id == model_configs.file_id)).first()
        if image_prop:
            effective_image_batch = model_configs.batch_size or image_prop.batch_size or 32
            data[DATASET].update(
                {
                    IMG_SIZE: image_prop.image_size,
                    BATCH_SIZE: effective_image_batch,
                    COLOR_MODE: image_prop.color_mode,
                    LABEL_MODE: image_prop.label_mode,
                }
            )

    logger.debug("Generating code for model type: %s", model_configs.model_type)
    template_loader = FileSystemLoader(searchpath=TEMPLATE_ROOT)
    template_env = Environment(loader=template_loader)
    template = template_env.get_template(_map_template(model_configs.model_type))

    rendered = template.render(data=data)
    if getattr(model_configs, "class_weight_mode", None) == "balanced":
        rendered = _inject_class_weights(
            code=rendered,
            problem_type_id=model_configs.model_type,
            epochs=model_configs.epochs,
            batch_size=model_configs.batch_size or 32,
        )
    return rendered


def _inject_class_weights(code: str, problem_type_id: int | None, epochs: int, batch_size: int) -> str:
    """Inject balanced class weighting calculation and model.fit class_weight parameter into generated code."""
    if problem_type_id == ProblemType.CLASSIFICATION:
        target_pattern = (
            f"    history = model.fit(\n        x_training, y_training, epochs={epochs}, batch_size={batch_size}\n    )"
        )
        replacement = (
            "    # Compute balanced class weights: w_c = N_samples / (n_classes * count(y == c))\n"
            "    y_raw = y_training.values if hasattr(y_training, 'values') else np.asarray(y_training)\n"
            "    if y_raw.ndim == 2:\n"
            "        classes, counts = np.unique(np.argmax(y_raw, axis=1), return_counts=True)\n"
            "    else:\n"
            "        classes, counts = np.unique(y_raw, return_counts=True)\n"
            "    n_samples = len(y_raw)\n"
            "    n_classes = len(classes)\n"
            "    class_weights = {\n"
            "        int(c): float(n_samples / (n_classes * count)) for c, count in zip(classes, counts, strict=True)\n"
            "    }\n"
            "\n"
            "    history = model.fit(\n"
            "        x_training,\n"
            "        y_training,\n"
            f"        epochs={epochs},\n"
            f"        batch_size={batch_size},\n"
            "        class_weight=class_weights,\n"
            "    )"
        )
        if target_pattern in code:
            return code.replace(target_pattern, replacement, 1)

    if problem_type_id == ProblemType.IMAGE_CLASSIFICATION:
        target_pattern = f"    history = model.fit(train_data, epochs={epochs})"
        replacement = (
            "    import os\n"
            "\n"
            "    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}\n"
            "    subdirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])\n"
            "    class_counts = [\n"
            "        sum(\n"
            "            1 for f in os.listdir(os.path.join(directory, s))\n"
            "            if os.path.splitext(f)[1].lower() in valid_exts\n"
            "        )\n"
            "        for s in subdirs\n"
            "    ]\n"
            "    total_samples = sum(class_counts)\n"
            "    n_classes = len(subdirs)\n"
            "    class_weights = {\n"
            "        i: float(total_samples / (n_classes * count)) for i, count in enumerate(class_counts)\n"
            "    }\n"
            "\n"
            "    history = model.fit(\n"
            "        train_data,\n"
            f"        epochs={epochs},\n"
            "        class_weight=class_weights,\n"
            "    )"
        )
        if target_pattern in code:
            return code.replace(target_pattern, replacement, 1)

    return code


def _map_template(problem_type_id: int | None) -> str:
    """Map a ProblemType enum value to its Jinja2 template path."""
    options = {
        ProblemType.CLASSIFICATION: CODE_TEMPLATE_FOLDER + "multi-class-all-float-classification-csv.py",
        ProblemType.REGRESSION: CODE_TEMPLATE_FOLDER + "linear-regression-all-float.py",
        ProblemType.IMAGE_CLASSIFICATION: CODE_TEMPLATE_FOLDER + "simple-image-classification.py",
    }
    try:
        return options[problem_type_id]
    except KeyError:
        valid = ", ".join(str(int(p)) for p in ProblemType)
        raise ValueError(f"Unknown model type: {problem_type_id!r}. Valid: {valid}") from None


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
