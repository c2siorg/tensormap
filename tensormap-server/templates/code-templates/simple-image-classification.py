import json
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

def deep_learning_model():
    image_size = ({{data.dataset.image_size}}, {{data.dataset.image_size}})
    batch_size = {{data.dataset.batch_size}}
    color_mode = '{{data.dataset.color_mode}}'
    label_mode = '{{data.dataset.label_mode}}'

    directory = '{{data.dataset.file_name}}'
    validation_split = 1 - ({{data.dataset.training_split}} / 100)
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode=label_mode
    )
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode=label_mode
    )

    json_string = json.dumps(yaml.load(open('{{data.dl_model.json_file}}')))
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)

    model.compile(
        optimizer='{{data.dl_model.optimizer}}',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['{{data.dl_model.metric}}'],
    )

    history = model.fit(train_data, epochs={{data.dl_model.epochs}})

    test_loss, test_acc = model.evaluate(test_data, verbose=2)

    return history, test_loss, test_acc

print('Starting')
history, test_loss, test_acc = deep_learning_model()
print('Finish')