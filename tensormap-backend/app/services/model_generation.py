"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
import os
from collections import defaultdict
import tensorflow as tf
from app.shared.logging_config import get_logger
from app.shared.constants import TEMPLATE_ROOT

logger = get_logger(__name__)

def _get_server_registry() -> dict:
    """Securely load the Single Source of Truth from the server disk."""
    registry_path = os.path.join(TEMPLATE_ROOT, "layer_registry.json")
    try:
        with open(registry_path, "r") as f:
            return json.load(f).get("layers", {})
    except Exception as e:
        logger.error(f"Failed to load server registry: {e}")
        return {}

def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure."""
    try:
        logger.debug("Generating model from %d nodes, %d edges", len(model_params.get("nodes", [])), len(model_params.get("edges", [])))
        server_registry = _get_server_registry()

        source_to_targets = defaultdict(list)
        target_to_sources = defaultdict(list)
        for edge in model_params.get("edges", []):
            source_to_targets[edge["source"]].append(edge["target"])
            target_to_sources[edge["target"]].append(edge["source"])

        nodes_by_id = {node["id"]: node for node in model_params.get("nodes", [])}
        keras_tensors = {}
        visited = set()
        queue = []
        input_tensors = []

        # 1. Dynamically find and instantiate Input nodes securely
        for node in model_params.get("nodes", []):
            data = node.get("data", {})
            client_display_name = data.get("registry", {}).get("display_name")
            
            # Backward compatibility check
            is_legacy_input = node.get("type") == "custominput"
            is_registry_input = client_display_name == "Input"

            if is_legacy_input or is_registry_input:
                params = data.get("params", {})
                dims = []
                for i in range(1, 4):
                    val = params.get(f"dim_{i}") or params.get(f"dim-{i}")
                    if val:
                        dims.append(int(val))
                
                # FIX: Hard validation error instead of silent fallback
                if not dims:
                    raise ValueError(f"Input dimensions are missing for Input node '{node['id']}'.")

                tensor = tf.keras.Input(shape=tuple(dims), name=node["id"])
                keras_tensors[node["id"]] = tensor
                input_tensors.append(tensor)
                visited.add(node["id"])
                queue.append(node["id"])

        if not queue:
            raise ValueError("FATAL: No Input layer was detected! BFS cannot start.")

        # 2. BFS from input nodes to build hidden layers
        while queue:
            current_id = queue.pop(0)
            for target_id in source_to_targets.get(current_id, []):
                if target_id in visited:
                    continue
                
                all_sources = target_to_sources.get(target_id, [])
                if not all(src in visited for src in all_sources):
                    continue

                visited.add(target_id)
                queue.append(target_id)

                source_tensors = [keras_tensors[src] for src in all_sources]
                if len(source_tensors) > 1:
                    input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
                else:
                    input_tensor = source_tensors[0]

                node = nodes_by_id[target_id]
                keras_tensors[target_id] = _build_secure_layer(node, input_tensor, server_registry)

        # 3. Identify outputs and serialize
        output_ids = [n["id"] for n in model_params.get("nodes", []) if n["id"] not in source_to_targets]
        
        outputs = []
        for oid in output_ids:
            if oid not in keras_tensors:
                raise KeyError(f"Output node '{oid}' was skipped! The graph might be disconnected.")
            outputs.append(keras_tensors[oid])

        model = tf.keras.Model(inputs=input_tensors, outputs=outputs)
        return json.loads(model.to_json())

    except Exception as e:
        # FIX: Replaced print/traceback with production-safe logger
        logger.exception("Crash detected in model generation.")
        raise e

def _build_secure_layer(node: dict, input_tensor, server_registry: dict):
    """Securely instantiate a Keras layer using only server-side configuration."""
    params = node.get("data", {}).get("params", {})
    client_display_name = node.get("data", {}).get("registry", {}).get("display_name")
    node_type = node.get("type")
    name = node["id"]

    # 1. Map client request to a trusted Server-Side configuration
    server_layer_config = None
    
    # Try mapping by the new dynamic display name
    for key, config in server_registry.items():
        if config.get("display_name") == client_display_name:
            server_layer_config = config
            break
            
    # Try backward compatibility mapping for old database models
    if not server_layer_config:
        legacy_map = {"customdense": "Dense", "customconv": "Conv2D", "customflatten": "Flatten"}
        if node_type in legacy_map:
            for key, config in server_registry.items():
                if config.get("display_name") == legacy_map[node_type]:
                    server_layer_config = config
                    break

    if not server_layer_config:
        raise ValueError(f"Untrusted or unknown layer requested: {client_display_name or node_type}")

    # 2. Extract trusted class name
    keras_mapping = server_layer_config.get("keras_mapping")
    class_name = keras_mapping.split(".")[-1]
    
    try:
        layer_class = getattr(tf.keras.layers, class_name)
    except AttributeError:
        raise ValueError(f"Invalid Keras layer class mapped on server: {class_name}")

    # 3. Securely inject parameters (Only allow parameters defined in the server JSON)
    allowed_params = server_layer_config.get("params", {})
    kwargs = {"name": name}
    
    for key, value in params.items():
        if key in allowed_params: # Security Check: Is this a permitted parameter?
            if isinstance(value, str) and "," in value:
                kwargs[key] = tuple(int(x.strip()) for x in value.split(","))
            else:
                kwargs[key] = value

    return layer_class(**kwargs)(input_tensor)