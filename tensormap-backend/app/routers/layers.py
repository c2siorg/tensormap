"""Layers router — exposes the layer registry to the frontend."""

from fastapi import APIRouter

from app.layers.registry import get_layer_schema

router = APIRouter(tags=["layers"])


@router.get("/layers/schema")
def get_layers():
    """Return all registered layer types with their metadata."""
    return {"success": True, "data": get_layer_schema()}
