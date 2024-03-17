import tensorflow as tf
import json
import pandas as pd
import numpy as np
import yaml

from endpoints.DeepLearning.models import ModelBasic
from endpoints.DataUpload.models import DataFile
from shared.constants import *
from shared.services.config import get_configs
from shared.utils import get_socket_ref

socketio = get_socket_ref()

class CustomProgressBar(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CustomProgressBar, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        model_result(f"Epoch {epoch+1}/{self.params['epochs']}", 0)

    def on_batch_end(self, batch, logs=None):
        if self.params['steps']:
            progress = batch / self.params['steps']
        else:
            progress = 0
        progress_bar_width = 50
        arrow = '>' * int(progress * progress_bar_width)
        spaces = '=' * (progress_bar_width - len(arrow))
        if 'mse' in logs:
            metric = f"MSE: {logs['mse']:.4f}"
        elif 'accuracy' in logs:
            metric = f"Accuracy: {logs['accuracy']:.4f}"
        else:
            metric = ""

        model_result(f"{batch+1}/{self.params['steps']}  [{arrow}{spaces}] {int(progress * 100)}% - Loss: {logs['loss']:.4f} - {metric}", 1)

    def on_test_begin(self, logs=None):
        model_result("Evaluating...", 2)

    def on_test_end(self, logs=None):
        model_result(f"Evaluation Results: Accuracy - {logs['accuracy']:.4f} Loss - {logs['loss']:.4f}", 3)
        model_result("Finish", 4)

def model_result(message, test):
    message = message.split('')[-1]
    data = {
        "message": message,
        "test": test
    }
    socketio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE)
    socketio.sleep(0)

def helper_generate_file_location(file_id):
    configs = get_configs()
    file = DataFile.query.filter_by(id=file_id).first()
    return configs['api']['upload']['folder'] + '/' + file.file_name + '.' + file.file_type

def helper_generate_json_model_file_location(model_name):
    return MODEL_GENERATION_LOCATION + model_name + MODEL_GENERATION_TYPE

def model_run(incoming):
    model_name = incoming[MODEL_NAME]
    model_configs = ModelBasic.query.filter_by(model_name=model_name).first()

    FILE_NAME = helper_generate_file_location(file_id=getattr(model_configs,FILE_ID))
    features = pd.read_csv(FILE_NAME)
    features.dropna(inplace=True)

    X = features.drop(getattr(model_configs,FILE_TARGET), axis=1)
    y = features[getattr(model_configs,FILE_TARGET)]

    split_index = int(len(X) * getattr(model_configs,MODEL_TRAINING_SPLIT) / 100)
    print(getattr(model_configs,MODEL_TRAINING_SPLIT))
    x_training = X[:split_index]
    y_training = y[:split_index]
    x_testing = X[split_index:]
    y_testing = y[split_index:]

    json_string = json.dumps(yaml.load(open(helper_generate_json_model_file_location(model_name=model_name))))
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)
    if getattr(model_configs,MODEL_LOSS) == 'sparse_categorical_crossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()
    model.compile(
        optimizer=getattr(model_configs,MODEL_OPTIMIZER),
        loss=loss,
        metrics=[getattr(model_configs,MODEL_METRIC)],
    )

    print(x_testing.shape, y_testing.shape)
    print(x_training.shape, y_training.shape)
    model.fit(x_training, y_training, epochs = getattr(model_configs,MODEL_EPOCHS),callbacks=[CustomProgressBar()], verbose=0)
    # model.fit(x_training, y_training, validation_data=(x_testing, y_testing), epochs = getattr(model_configs,MODEL_EPOCHS))
    model.evaluate(x_testing, y_testing, callbacks=[CustomProgressBar()], verbose=0)