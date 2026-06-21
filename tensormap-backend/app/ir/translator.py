from collections import defaultdict

from pydantic import ValidationError

from app.ir.schema import (
    AvgPool2DParams,
    BatchNormParams,
    ConcatenateParams,
    Conv2DParams,
    DenseParams,
    DropoutParams,
    EmbeddingParams,
    FlattenParams,
    GlobalAvgPool2DParams,
    GRUParams,
    InputParams,
    IREdge,
    IRGraph,
    IRNode,
    LSTMParams,
    MaxPool2DParams,
    NodeParams,
    ReshapeParams,
    SimpleRNNParams,
)


class TranslationError(Exception):
    def __init__(self, message, field_name=None):
        super().__init__(message)
        self.field_name = field_name


_PARAM_MODELS = {
    "input": InputParams,
    "dense": DenseParams,
    "flatten": FlattenParams,
    "conv2d": Conv2DParams,
    "maxpool2d": MaxPool2DParams,
    "avgpool2d": AvgPool2DParams,
    "globalavgpool2d": GlobalAvgPool2DParams,
    "lstm": LSTMParams,
    "gru": GRUParams,
    "simplernn": SimpleRNNParams,
    "embedding": EmbeddingParams,
    "dropout": DropoutParams,
    "batchnorm": BatchNormParams,
    "reshape": ReshapeParams,
    "concatenate": ConcatenateParams,
}


def translate_params_to_ir(layer_type: str, raw_params: dict) -> NodeParams:
    """Validates raw params and returns typed NodeParams. Can raise TranslationError.

    This function handles both new IRGraph format and legacy ReactFlow format.
    """
    if layer_type not in _PARAM_MODELS:
        raise TranslationError(f"Unknown layer type: {layer_type}")

    ModelClass = _PARAM_MODELS[layer_type]

    # Handle legacy format conversions
    params_to_validate = dict(raw_params)
    params_to_validate["layer_type"] = layer_type

    # Legacy conversions for specific layer types
    if layer_type == "input" and "shape" not in params_to_validate and "dim-1" in params_to_validate:
        # Convert old dim-1 format to shape
        params_to_validate["shape"] = int(params_to_validate["dim-1"])

    if layer_type == "conv2d":
        # Handle legacy format: kernelX/kernelY → kernel_size, filter → filters
        if "filter" in params_to_validate and "filters" not in params_to_validate:
            params_to_validate["filters"] = params_to_validate["filter"]
        if "kernelX" in params_to_validate and "kernel_size" not in params_to_validate:
            params_to_validate["kernel_size"] = params_to_validate["kernelX"]

    if (
        (layer_type == "maxpool2d" or layer_type == "avgpool2d")
        and "stride" in params_to_validate
        and "strides" not in params_to_validate
    ):
        # Handle legacy format: stride → strides
        params_to_validate["strides"] = params_to_validate["stride"]

    try:
        # Pydantic will validate
        params = ModelClass(**params_to_validate)
        return params
    except ValidationError as e:
        # Extract field-level validation errors from Pydantic
        field_errors = []
        for error in e.errors():
            field_name = ".".join(str(loc) for loc in error["loc"])
            field_errors.append(f"{field_name}: {error['msg']}")
        error_msg = f"Validation failed for {layer_type}: {'; '.join(field_errors)}"
        # Use first field name for field_name attribute
        first_field = str(e.errors()[0]["loc"][0]) if e.errors() else None
        raise TranslationError(error_msg, field_name=first_field) from e
    except Exception as e:
        raise TranslationError(f"Unexpected error validating {layer_type}: {str(e)}") from e


def reactflow_to_ir(canvas_json: dict) -> IRGraph:
    nodes = []
    edges = []

    rf_edges = canvas_json.get("edges", [])
    # Build incoming edges map for Concatenate order
    in_edges_map = defaultdict(list)
    for e in rf_edges:
        in_edges_map[e.get("target")].append(e)

    for n in canvas_json.get("nodes", []):
        ndata = n.get("data", {})
        nparams = ndata.get("params", {})
        ntype = n.get("type", "").lower().replace("node", "")

        # Handle legacy "custom" prefix (e.g., "custominput" → "input")
        if ntype.startswith("custom"):
            ntype = ntype[6:]  # Remove "custom" prefix

        # old format might map ntype directly if we align it
        if not ntype and "layer_type" in nparams:
            ntype = nparams["layer_type"]
        elif not ntype:
            raise TranslationError(f"Node {n.get('id')}: Unable to determine layer type from node type or params")

        try:
            typed_params = translate_params_to_ir(ntype, nparams)
        except TranslationError as e:
            # Re-raise with node context for better error messages
            raise TranslationError(f"Node {n.get('id')}: {e}", field_name=e.field_name) from e

        pos = n.get("position", {"x": 0, "y": 0})

        # Determine input node ids
        in_nodes = [e.get("source") for e in in_edges_map.get(n.get("id"), [])]

        ir_n = IRNode(
            id=n.get("id"),
            node_params=typed_params,
            name=ndata.get("name", ntype),
            position=pos,
            input_node_ids=in_nodes,
        )
        nodes.append(ir_n)

    for e in rf_edges:
        edges.append(
            IREdge(
                id=e.get("id", f"{e.get('source')}-{e.get('target')}"),
                source_id=e.get("source"),
                target_id=e.get("target"),
                source_handle=e.get("sourceHandle"),
                target_handle=e.get("targetHandle"),
            )
        )

    return IRGraph(version="1.0", nodes=nodes, edges=edges)


def ir_to_keras_build_args(node: IRNode) -> dict:
    """Returns kwargs for Keras layer constructor from typed IR params."""
    params = node.node_params.model_dump()
    params.pop("layer_type", None)

    # Special handling for Input layer: convert shape int to tuple
    if node.node_params.layer_type == "input" and "shape" in params:
        # Keras Input expects shape as tuple, e.g., (784,)
        params["shape"] = (params["shape"],)

    # Special handling for Reshape layer: convert target_shape string to tuple
    if node.node_params.layer_type == "reshape" and "target_shape" in params:
        # Convert "7,7,64" → (7, 7, 64)
        shape_str = params["target_shape"]
        params["target_shape"] = tuple(int(x.strip()) for x in shape_str.split(","))

    if node.node_params.layer_type == "dropout" and "rate" in params:
        # test_ir_build_args_dropout expects {"rate": 0.5}
        return {"rate": params["rate"]}

    return params
