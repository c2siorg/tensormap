"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
import traceback
from collections import defaultdict
import tensorflow as tf
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure."""
    try:
        # --- X-RAY: PRINT INCOMING GRAPH FROM REACT ---
        print("\n" + "="*50)
        print("🔍 X-RAY: INCOMING GRAPH NODES 🔍")
        for node in model_params.get("nodes", []):
            print(f"[{node.get('type')}] ID: {node.get('id')}")
            print(f"  -> Data: {json.dumps(node.get('data', {}))}")
        print("="*50 + "\n")

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

        # 1. Dynamically find and instantiate Input nodes
        for node in model_params.get("nodes", []):
            data = node.get("data", {})
            registry = data.get("registry", {})
            
            # Robustly check if this is an Input layer
            is_input = (
                registry.get("display_name") == "Input" or 
                "InputLayer" in registry.get("keras_mapping", "") or
                node.get("type") == "custominput"
            )

            if is_input:
                params = data.get("params", {})
                
                # Extract dimensions safely
                dims = []
                for i in range(1, 4):
                    val = params.get(f"dim_{i}") or params.get(f"dim-{i}")
                    if val:
                        dims.append(int(val))
                
                # Fallback so Keras doesn't crash if UI sends nothing
                if not dims:
                    dims = [28, 28, 1] 

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
                keras_tensors[target_id] = _build_layer(node, input_tensor)

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
        print("\n" + "="*60)
        print("💥 CRASH DETECTED IN MODEL GENERATION 💥")
        traceback.print_exc()
        print("="*60 + "\n")
        raise e

def _build_layer(node: dict, input_tensor):
    """Instantiate a Keras layer dynamically using the registry."""
    params = node.get("data", {}).get("params", {})
    registry = node.get("data", {}).get("registry", {})
    keras_mapping = registry.get("keras_mapping")
    name = node["id"]

    if not keras_mapping:
        raise ValueError(f"Node '{name}' missing keras_mapping. Is it an old graph?")

    class_name = keras_mapping.split(".")[-1]
    layer_class = getattr(tf.keras.layers, class_name)

    kwargs = {"name": name}
    for key, value in params.items():
        if isinstance(value, str) and "," in value:
            kwargs[key] = tuple(int(x.strip()) for x in value.split(","))
        else:
            kwargs[key] = value

    return layer_class(**kwargs)(input_tensor)