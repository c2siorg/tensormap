# Graph analysis

`app/ir/analysis` checks a model graph **without building it in TensorFlow**. It walks the graph in
topological order and works out, for every layer:

- the output shape (batch axis excluded, e.g. `(26, 26, 8)`),
- the parameter count, split into trainable and non-trainable,
- the multiply-accumulates for one sample,

and reports every problem as a structured diagnostic that names the layer causing it. It imports no
TensorFlow, so analysing a small CNN takes about 0.1 ms and it can run on every canvas edit.

## Why it exists

Before this module, the first time a user learned that a Conv2D was fed a flat vector was when a
Keras call failed deep inside model generation. There was also no way to show shapes while drawing, and
`_estimate_param_count` guessed (it assumed the previous layer's width was the Conv2D's input channels
and ignored graph topology).

## Using it

```python
from app.ir.analysis import analyze_graph, analyze_canvas

analysis = analyze_graph(ir_graph)          # a validated IRGraph
analysis = analyze_canvas(reactflow_json)   # raw canvas; invalid nodes become diagnostics

analysis.ok                       # no error-level diagnostics
analysis.node("conv1").output_shape   # (26, 26, 8)
analysis.total_params             # 13,610
for d in analysis.errors():
    print(d.code, d.node_id, d.message, d.suggestion)
```

Over HTTP: `POST /api/v1/layers/analyze-graph` with `{"canvas": {...}}` or `{"graph_ir": {...}}`.
It always answers 200; an invalid graph is a result, not a request failure.

```json
{
  "ok": false,
  "diagnostics": [{
    "code": "rank_mismatch",
    "severity": "error",
    "node_id": "conv1",
    "message": "Conv2D 'conv1': expects a 3-D input (height, width, channels) but receives a 1-D input (784,).",
    "suggestion": "Insert a Reshape layer with target shape '28,28,1' before this layer.",
    "related_node_ids": ["in"]
  }]
}
```

The canvas shows the result live: each layer displays its inferred output shape and parameter count,
flagged layers are outlined, and the "Model check" panel under the canvas lists the diagnostics with a
link that focuses the layer.

## Behaviour worth knowing

- **One mistake, one diagnostic.** A node whose inputs are unknown (because something upstream failed) is
  skipped silently, so a bad Reshape does not produce a message on every layer after it.
- **Edit-time tolerance.** `analyze_canvas` translates nodes one at a time. A node with missing or invalid
  settings gets an `invalid_params` diagnostic and the rest of the graph is still analysed.
- **Several inputs to one layer.** Merge layers (`Concatenate`, `Add`) expect multiple incoming connections.
  `Concatenate` joins tensors along an axis (defaulting to the last axis; `axis` follows Keras and counts the batch axis, so `1` is the first feature axis and `0` is rejected), while `Add` performs element-wise addition
  requiring strictly identical input shapes (ideal for ResNet-style skip/residual connections). Other layers that
  receive multiple connections merge them along the last axis as the Keras generator does.
- **Warnings are not errors.** `unflattened_dense` (Dense after Conv2D without Flatten) builds fine in Keras 3
  and is only flagged, because it is usually not what the user meant.
- **Defensive canvas normalization.** `analyze_canvas` handles null values, malformed nodes, missing IDs,
  and incomplete connections gracefully without raising unhandled server errors.
- **Output leaf nodes.** Only valid layers that have a successfully inferred output shape and are reachable
  from an Input layer are included in `output_node_ids`, matching Keras model generation semantics.
- **Preserved analysis on network errors.** The canvas hook preserves the last successful analysis if a
  temporary network disconnection occurs while sketching.
- **Multi-input and multi-output architectures.** Graphs can have multiple Input layers (e.g., multi-modal inputs
  or separate feature branches) and multiple terminal output layers (e.g., multi-task heads). Outputs are
  surfaced in `output_node_ids`. When a diagnostic involves multiple layers (such as all nodes in a `cycle` or
  the incoming branches of a `shape_mismatch`), they are exposed in `related_node_ids`, highlighted on the canvas,
  and presented as direct navigation buttons in the Model check panel.

## Diagnostic codes

| Code | Severity | Meaning |
|------|----------|---------|
| `empty_graph` | error | No layers on the canvas |
| `no_input_node` | error | No Input layer |
| `no_output_node` | error | No terminal output layer downstream of an Input layer |
| `disconnected_input` | error | An Input layer has no outgoing connection to downstream layers |
| `invalid_params` | error | A layer's type or settings do not validate |
| `dangling_edge` | error | A connection points at a layer that does not exist or has missing endpoints |
| `cycle` | error | The graph is not acyclic; `related_node_ids` lists the layers on the cycle |
| `disconnected_node` | error | A non-input layer has no incoming connection |
| `merge_arity` | error | A merge layer (Concatenate, Add) has fewer than two inputs |
| `rank_mismatch` | error | A layer needs a different number of dimensions than it receives |
| `shape_mismatch` | error | Inputs cannot be merged (axis mismatch or differing shapes for Add) |
| `invalid_output_size` | error | A `valid` kernel or pooling window is larger than the input |
| `invalid_reshape` | error | Reshape target is malformed or its size does not match the input |
| `duplicate_node_id` | error | Multiple layers share the same identifier |
| `duplicate_edge` | error | Multiple connections exist between the same source and target layers |
| `self_loop` | error | A layer connects directly into itself |
| `input_has_incoming_edge` | warning | Connections into an Input layer are ignored |
| `unflattened_dense` | warning | Dense applied to a 3-D or higher tensor keeps its rank |

Codes are stable identifiers; match on them, not on message text.

## Adding a layer

1. Add the layer to `app/layers/registry.py` and `app/ir/schema.py` as before.
2. Add a rule in `app/ir/analysis/shapes.py`:

   ```python
   @shape_rule("my_layer")
   def _my_layer(p: MyLayerParams, inputs: list[Shape]) -> LayerResult:
       (shape,) = inputs                      # merge layers receive several shapes
       return LayerResult((*shape[:-1], p.units), params=..., macs=...)
   ```

   Raise `ShapeError(code, detail, suggestion)` when the inputs are unacceptable.
3. Add a graph using it to `GRAPHS` in `tests/test_graph_analysis_parity.py`.

`test_every_registry_layer_has_a_rule` fails until step 2 is done, and the parity test fails if the rule
disagrees with Keras.

## How it is tested

- `tests/test_graph_analysis.py`: unit tests for every rule and diagnostic, canvas tolerance, and the endpoint.
- `tests/test_graph_analysis_parity.py`: builds each graph with the real `TensorFlowGenerator` and compares
  output shapes and parameter counts layer by layer, then checks that graphs the analyzer rejects also fail
  to build in Keras. This is what keeps the pure-Python rules honest.

## Not covered yet

- Input is a single integer (`shape: int`) in the IR, so image inputs are drawn as a flat vector plus a
  Reshape. Multi-axis inputs need a schema change and a migration of stored graphs.
- Dynamic axes (variable sequence length) are not modelled; every dimension is known.
- Memory estimates cover weights only, not activations or optimizer state.
- The legacy `model_generation()` path is still the fallback when the registry-driven generator fails; the
  analyzer is the first step towards retiring it, since it already covers every registry layer.
