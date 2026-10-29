# TensorMap API Changelog

## v1 (Current)

### Data Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/data/upload/file | Upload a CSV file |
| GET | /api/v1/data/upload/file | List uploaded files |
| DELETE | /api/v1/data/upload/file/{file_id} | Delete a file |

### Data Process
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/data/process/file/{file_id} | Get file preview (paginated) |
| GET | /api/v1/data/process/file/{file_id}/columns | Get column names |
| GET | /api/v1/data/process/stats/{file_id} | Column statistics |
| GET | /api/v1/data/process/correlation/{file_id} | Correlation matrix |
| GET | /api/v1/data/process/data_metrics/{file_id} | Data metrics |
| POST | /api/v1/data/process/target | Set target column |
| GET | /api/v1/data/process/target | List all target assignments |
| GET | /api/v1/data/process/target/{file_id} | Get target for a file |
| DELETE | /api/v1/data/process/target/{file_id} | Delete target assignment |
| POST | /api/v1/data/process/preprocess/{file_id} | Apply transformations |

### Project
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/project | List projects |
| POST | /api/v1/project | Create project |
| GET | /api/v1/project/{project_id} | Get project details |
| PATCH | /api/v1/project/{project_id} | Update project |
| DELETE | /api/v1/project/{project_id} | Delete project |

### Model (Deep Learning)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/model/training-history | List training history |
| GET | /api/v1/model/model-list | List models |
| POST | /api/v1/model/validate | Validate model graph |
| POST | /api/v1/model/save | Save model architecture |
| PATCH | /api/v1/model/training-config | Update training config |
| GET | /api/v1/model/{model_name}/graph | Get model graph |
| DELETE | /api/v1/model/{model_id} | Delete model |
| POST | /api/v1/model/code | Generate training code |
| POST | /api/v1/model/run | Run model training |

### Model analysis exports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/model/analysis/{job_id}/export?format=json | Download available cached analysis results as JSON (default) |
| GET | /api/v1/model/analysis/{job_id}/export?format=csv | Download available cached analysis results as flattened CSV |

Exports require a completed training job. Only cached analysis output for that job and task type is included; this endpoint does not start or repeat analysis. JSON has `job_id`, `analysis_type`, `results` (keyed by analysis cache result name), and `unavailable_results` (supported outputs not yet cached). CSV columns are `analysis,field,type,value`. `field` is a JSON Pointer path: the empty string identifies the root; other paths start with `/`, `/` separates segments, `~1` encodes `/`, and `~0` encodes `~`. Array indexes are decimal path segments, interpreted as indexes when the parent row has type `array`; this parent `type` marker also distinguishes arrays from objects with numeric keys. Each object/array has a row with type `object` or `array`; non-empty containers have an empty `value`, while empty containers have JSON values `{}` or `[]`. Scalar rows have type `scalar` and a JSON-encoded value (`null`, `""`, strings, numbers, or booleans). Thus missing fields have no row, while null and empty strings remain distinct. Decode path escapes in the order `~1` to `/`, then `~0` to `~`. A completed job with no available cached results returns 409; missing jobs return 404, incomplete jobs return 400, and unsupported format values return 422. Image-classification exports omit outputs that are not supported for that task; regression exports omit classification-only outputs.

### Layers and graph analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/layers | Layer registry grouped by category |
| GET | /api/v1/layers/{type_key} | One layer spec |
| POST | /api/v1/layers/validate-graph | Structural validation of an IR graph |
| POST | /api/v1/layers/analyze-graph | Static analysis: per-layer output shapes, parameter counts and diagnostics (see [GRAPH_ANALYSIS.md](GRAPH_ANALYSIS.md)) |

`analyze-graph` takes `{"graph_ir": {...}}` or `{"canvas": {"nodes": [...], "edges": [...]}}` and always returns
200 with the analysis; an invalid graph is reported in `diagnostics`. It returns 400 when neither key is present
and 422 when `graph_ir` does not parse. It does not use the `{success, message, data}` envelope.

## Response Format

```json
{
  "success": true,
  "message": "Done",
  "data": { }
}
```

## Status Codes

- 200: Success
- 201: Created
- 400: Bad Request
- 404: Not Found
- 413: Payload Too Large (file uploads)
- 422: Validation Error
- 500: Server Error

## Pagination

Use `page` and `page_size` query params. Default page_size: 50, max page_size: 1000.

## WebSocket Events

Namespace: `/dl-result`

- `result` - Training progress and results

---

*Generated from FastAPI routes*
