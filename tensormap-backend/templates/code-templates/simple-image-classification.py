import tensorflow as tf


def deep_learning_model():
    image_size = ({{data.dataset.image_size}}, {{data.dataset.image_size}})
    batch_size = {{data.dataset.batch_size}}
    color_mode = "{{data.dataset.color_mode}}"
    label_mode = "{{data.dataset.label_mode}}"

    directory = "{{data.dataset.file_name}}"
    validation_split = 1 - ({{data.dataset.training_split}} / 100)
    train_data = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode=label_mode,
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode=label_mode,
    )

    with open("{{data.dl_model.json_file}}") as f:
        json_string = f.read()
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)

    model.compile(
        optimizer="{{data.dl_model.optimizer}}",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["{{data.dl_model.metric}}"],
    )

    class_weight_mode = "{{data.dl_model.class_weight_mode}}"
    class_weights = None
    if class_weight_mode == "balanced":
        import numpy as np

        y_train_list = [y.numpy() for _, y in train_data]
        if y_train_list:
            y_train = np.concatenate(y_train_list, axis=0)
            if y_train.ndim > 1:
                y_train = np.argmax(y_train, axis=1) if y_train.shape[1] > 1 else y_train.ravel()
            classes, counts = np.unique(y_train, return_counts=True)
            n_samples = float(len(y_train))
            n_classes = float(len(classes))
            class_weights = {
                int(c): float(n_samples / (n_classes * count))
                for c, count in zip(classes, counts, strict=False)
                if count > 0
            }

    history = model.fit(train_data, epochs={{data.dl_model.epochs}}, class_weight=class_weights)

    test_loss, test_acc = model.evaluate(test_data, verbose=2)

    return history, test_loss, test_acc


print("Starting")
history, test_loss, test_acc = deep_learning_model()
print("Finish")
