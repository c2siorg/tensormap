import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.config import get_settings
from app.models.data import DataFile, ImageProperties
from app.models.ml import ModelBasic
from app.shared.constants import (
    MODEL_GENERATION_LOCATION,
    MODEL_GENERATION_TYPE,
    SOCKETIO_DL_NAMESPACE,
    SOCKETIO_LISTENER,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)


class CustomProgressBar(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        _model_result(f"Epoch {epoch + 1}/{self.params['epochs']}", 0)

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        if self.params["steps"]:
            progress = batch / self.params["steps"]
        else:
            progress = 0
        progress_bar_width = 50
        arrow = ">" * int(progress * progress_bar_width)
        spaces = "=" * (progress_bar_width - len(arrow))
        if "mse" in logs:
            metric = f"MSE: {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy: {logs['accuracy']:.4f}"
        else:
            metric = ""

        _model_result(
            f"{batch + 1}/{self.params['steps']}  [{arrow}{spaces}] "
            f"{int(progress * 100)}% - Loss: {logs['loss']:.4f} - {metric}",
            1,
        )

    def on_test_begin(self, logs: dict = None) -> None:
        _model_result("Evaluating...", 2)

    def on_test_end(self, logs: dict = None) -> None:
        if "mse" in logs:
            metric = f"MSE Loss- {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy- {logs['accuracy']:.4f}"
        else:
            metric = ""
        _model_result(f"Evaluation Results: {metric} Loss - {logs['loss']:.4f}", 3)
        _model_result("Finish", 4)


def _model_result(message: str, test: int) -> None:
    import asyncio

    data = {"message": message, "test": test}
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE))
        else:
            loop.run_until_complete(sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE))


def _helper_generate_file_location(db: Session, file_id) -> str:
    upload_folder = get_settings().upload_folder
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file.file_type == "zip":
        return upload_folder + "/" + file.file_name
    return upload_folder + "/" + file.file_name + "." + file.file_type


def _helper_generate_json_model_file_location(model_name: str) -> str:
    return MODEL_GENERATION_LOCATION + model_name + MODEL_GENERATION_TYPE


def model_run(model_name: str, db: Session) -> None:
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        image_properties = db.exec(select(ImageProperties).where(ImageProperties.id == model_configs.file_id)).first()
        image_size = (image_properties.image_size, image_properties.image_size)
        batch_size = image_properties.batch_size
        color_mode = image_properties.color_mode
        label_mode = image_properties.label_mode

        directory = _helper_generate_file_location(db, file_id=model_configs.file_id)
        logger.debug(
            "Image classification params - directory: %s, image_size: %s, "
            "batch_size: %s, color_mode: %s, label_mode: %s",
            directory,
            image_size,
            batch_size,
            color_mode,
            label_mode,
        )
        validation_split = 1 - (model_configs.training_split / 100)
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
    else:
        file_location = _helper_generate_file_location(db, file_id=model_configs.file_id)
        features = pd.read_csv(file_location)
        features.dropna(inplace=True)

        X = features.drop(model_configs.target_field, axis=1)
        y = features[model_configs.target_field]

        split_index = int(len(X) * model_configs.training_split / 100)
        x_training = X[:split_index]
        y_training = y[:split_index]
        x_testing = X[split_index:]
        y_testing = y[split_index:]

    from werkzeug.utils import secure_filename
    s_model_name = secure_filename(model_name)
    with open(_helper_generate_json_model_file_location(model_name=s_model_name)) as f:
        # Prevent YAML arbitrary code execution (though json string might not need yaml load at all)
        # Using json.loads is safer since the data is a JSON string of keras model
        import json
        json_string = f.read()
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)
    model.summary()
    if model_configs.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()
    model.compile(
        optimizer=model_configs.optimizer,
        loss=loss,
        metrics=[model_configs.metric],
    )

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=model_configs.epochs,
            callbacks=[CustomProgressBar()],
            verbose=0,
        )
        model.evaluate(test_data, callbacks=[CustomProgressBar()], verbose=0)
    else:
        model.fit(
            x_training,
            y_training,
            epochs=model_configs.epochs,
            callbacks=[CustomProgressBar()],
            verbose=0,
        )
        model.evaluate(x_testing, y_testing, callbacks=[CustomProgressBar()], verbose=0)
