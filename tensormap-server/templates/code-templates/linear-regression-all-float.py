import json

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


def deep_learning_model():
    features = pd.read_csv('{{data.dataset.file_name}}')
    features.dropna(inplace=True)

    # Split Training and testing sets
    X = features.drop('{{data.dataset.target_field}}', axis=1)
    y = features['{{data.dataset.target_field}}']

    split_index = int(len(X) * {{data.dataset.training_split}} / 100)
    
    x_training = X[:split_index]
    y_training = y[:split_index]
    x_testing = X[split_index:]
    y_testing = y[split_index:]

    json_string = json.dumps(yaml.load(open('{{data.dl_model.json_file}}')))
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)

    model.compile(
        optimizer='{{data.dl_model.optimizer}}',
        loss='mean_squared_error',
        metrics=['{{data.dl_model.metric}}'],
    )

    history = model.fit(x_training, y_training, epochs={{data.dl_model.epochs}})

    test_loss = model.evaluate(x_testing, y_testing, verbose=2)

    return history, test_loss

print('Starting')
history, test_loss = linear_regression_model()
print('Finish')

