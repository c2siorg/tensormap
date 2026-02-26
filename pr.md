# feat: add full activation function options to Conv2D and Dense nodes

Closes #129

## Summary

Expands the activation function dropdown for the Conv2D node from the two hardcoded options (`none`, `relu`) to the full set of commonly used Keras activation functions: `none`, `relu`, `sigmoid`, `tanh`, `softmax`, `elu`, `selu`.

For consistency, the Dense node dropdown is unified to use the same list. A single shared `ACTIVATIONS` constant replaces the two separate `denseActivations` and `convActivations` arrays (DRY). The backend already maps `"none"` → `"linear"` for Conv2D; the same mapping is now applied to Dense so both layer types behave correctly when `"none"` is selected.

## Changes

### Root Cause
`convActivations` in `NodePropertiesPanel.jsx` only contained `["none", "relu"]`. The backend had no issues — it passes the activation string directly to `tf.keras.layers.Conv2D(activation=…)`, which natively supports all standard Keras names.

### Frontend
- **`NodePropertiesPanel.jsx`** — Replaced separate `denseActivations` and `convActivations` arrays with a single `ACTIVATIONS` constant containing all 7 options. Both the Dense and Conv2D `<Select>` dropdowns now render from this shared list.

### Backend
- **`model_generation.py`** — Added `"none"` → `"linear"` mapping for the `customdense` branch in `_build_layer()` to match the existing Conv2D handling.

### No changes needed
- **`ConvNode.jsx`** — already renders activation via `Act: ${p.activation}` and suppresses it when `"none"`.
- **`Canvas.jsx`** — Conv2D default activation is already `"none"`.

## Acceptance Criteria

- [x] Conv2D property panel shows all 7 activation options: none, relu, sigmoid, tanh, softmax, elu, selu
- [x] Dense property panel shows the same 7 activation options for consistency
- [x] Default activation for Conv2D remains `"none"`
- [x] Selecting each option correctly updates node data via `updateParam`
- [x] Backend maps `"none"` → `"linear"` for both Conv2D and Dense layers
- [x] No regressions — existing activation values (`relu`, `sigmoid`, etc.) still work
- [x] Conv2D node on canvas displays the selected activation name

## Changed Files

| File | Change |
|---|---|
| `tensormap-frontend/src/components/DragAndDropCanvas/NodePropertiesPanel.jsx` | Unified activation options into shared `ACTIVATIONS` constant (7 options) |
| `tensormap-backend/app/services/model_generation.py` | Map `"none"` → `"linear"` for Dense layer activation |

## How to Test

```bash
# Frontend — start dev server and verify dropdowns
cd tensormap-frontend
npm install
npm run dev
# Open canvas, add a Conv2D node, open properties panel — confirm 7 activation options appear
# Repeat for Dense node

# Backend — run existing tests
cd tensormap-backend
uv run pytest -v
```
