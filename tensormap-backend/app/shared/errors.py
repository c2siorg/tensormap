"""Mapping of internal TensorFlow/SQLAlchemy error substrings to user-facing messages.

These are used as a last-resort fallback in the service layer when a
ModelValidationError was *not* raised by model_generation() but an exception
still surfaces from tf.keras (e.g., during the JSON round-trip).  Add entries
here for any cryptic TF message that leaks through to users.
"""

err_msgs = {
    # Concatenate layer shape mismatch
    "A `Concatenate` layer requires inputs with matching shapes except for the concat axis.": (
        "Incompatible inputs for the Concatenate layer. "
        "All branches being merged must have the same shape except along the concatenation axis."
    ),
    # SQLAlchemy duplicate key
    "Duplicate entry": "Model name already exists.",
    # Keras model input/output incompatibility (running phase)
    "incompatible with layer model": (
        "Model running failed: the input or output shapes are incompatible with the saved model. "
        "Re-validate the model before training."
    ),
    # Generic shape mismatch
    "Incompatible shapes": (
        "Shape mismatch in the model. "
        "Check that consecutive layers have compatible dimensions "
        "(e.g., add a Flatten layer before a Dense layer when using Conv2D)."
    ),
    # tf.errors.InvalidArgumentError surface strings
    "Invalid argument": (
        "A layer received an invalid argument. "
        "Check all layer parameters (units, filters, kernel size, etc.) for correct values."
    ),
}
