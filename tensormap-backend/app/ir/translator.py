from collections import defaultdict

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
    """Validates raw params and returns typed NodeParams. Can raise TranslationError."""
    if layer_type not in _PARAM_MODELS:
        raise TranslationError(f"Unknown layer type: {layer_type}")

    ModelClass = _PARAM_MODELS[layer_type]
    try:
        # Pydantic will validate
        params = ModelClass(**raw_params)
        return params
    except Exception as e:
        # Simplified: realistically we can inspect Pydantic ValidationError
        raise TranslationError(str(e)) from e


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

    if node.node_params.layer_type == "dropout" and "rate" in params:
        # test_ir_build_args_dropout expects {"rate": 0.5}
        return {"rate": params["rate"]}

    return params
