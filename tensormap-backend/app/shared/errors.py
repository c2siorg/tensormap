"""Mapping of internal TensorFlow/SQLAlchemy error substrings to user-facing messages."""

err_msgs = {
    "A `Concatenate` layer requires inputs with matching shapes except for the concat axis.": (
        "Incompatible inputs for multi-input layers."
    ),
    "Duplicate entry": "Model name already exists",
    "incompatible with layer model": "MODEL RUNNING FAILED : Incompatible shapes for input or output layers",
    "Incompatible shapes": "Error in Model values",
}


def _expected_input_rank(layer) -> tuple[int | None, int | None]:
    """Return the (min, max) input rank a Keras layer accepts, read from its ``input_spec``."""
    spec = getattr(layer, "input_spec", None)
    if isinstance(spec, list | tuple):
        spec = spec[0] if len(spec) == 1 else None
    if spec is None:
        return None, None
    if spec.ndim is not None:
        return spec.ndim, spec.ndim
    return spec.min_ndim, spec.max_ndim


def layer_shape_mismatch_message(target_id: str, source_ids, input_shape, layer) -> str:
    """Explain why ``layer`` (node ``target_id``) rejected the tensor coming from ``source_ids``.

    The text is built from the incoming shape and the layer's declared
    ``input_spec`` rather than from the Keras exception, so it names the two
    nodes and what to change without repeating framework internals.
    """
    source = ", ".join(source_ids) if source_ids else "its input"
    shape = tuple(input_shape)
    rank = len(shape)
    min_rank, max_rank = _expected_input_rank(layer)
    prefix = f"Shape mismatch between {source} and {target_id}: "

    if min_rank is not None and rank < min_rank:
        example = " such as (height, width, channels)" if min_rank == 4 else ""
        return (
            f"{prefix}{target_id} expects a {min_rank}-D input but receives the {rank}-D output of {source} "
            f"with shape {shape}. Place {target_id} before any Flatten or Dense layer, or give the Input layer "
            f"a {min_rank - 1}-D shape{example}."
        )
    if max_rank is not None and rank > max_rank:
        return (
            f"{prefix}{target_id} expects a {max_rank}-D input but receives the {rank}-D output of {source} "
            f"with shape {shape}. Add a Flatten or Reshape layer between {source} and {target_id} so that it "
            f"receives {max_rank}-D data."
        )
    return (
        f"{prefix}{target_id} cannot be applied to the output of {source} with shape {shape}. "
        f"Check the parameters of {target_id} (kernel size, pool size, strides) against that shape."
    )
