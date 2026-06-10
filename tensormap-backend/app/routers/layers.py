from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.ir.schema import IRGraph, validate_ir_graph
from app.layers.registry import LAYER_CATEGORIES, LAYER_REGISTRY, get_layers_by_category

router = APIRouter(prefix="/layers", tags=["layers"])


@router.get("/")
async def get_layer_registry():
    """
    Returns the complete layer registry grouped by category.
    """
    layers_by_cat = get_layers_by_category()

    return {
        "categories": LAYER_CATEGORIES,
        "layers": {cat: [spec.to_api_dict() for spec in specs] for cat, specs in layers_by_cat.items()},
    }


@router.post("/validate-graph")
async def validate_graph_endpoint(body: dict[str, Any]):
    """
    Receives {"graph_ir": {...}}
    Deserializes to IRGraph -> validate_topology -> returns {valid: bool, errors: [...]}
    """
    if "graph_ir" not in body:
        raise HTTPException(status_code=400, detail="Missing 'graph_ir' in request body.")
    graph_ir_data = body["graph_ir"]

    try:
        graph = IRGraph(**graph_ir_data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e

    try:
        errors = validate_ir_graph(graph)
        if errors:
            return JSONResponse(status_code=400, content={"valid": False, "errors": [err.to_dict() for err in errors]})
        return {"valid": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{type_key}")
async def get_layer_spec(type_key: str):
    """
    Returns single LayerSpec as JSON.
    Returns 404 if type_key not found.
    """
    if type_key not in LAYER_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Layer type '{type_key}' not found.")
    return LAYER_REGISTRY[type_key].to_api_dict()
