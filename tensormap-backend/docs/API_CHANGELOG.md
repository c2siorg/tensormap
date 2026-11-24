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
| GET | /api/v1/data/process/quality/{file_id} | Dataset quality report |
| GET | /api/v1/data/process/class-imbalance/{file_id} | Target class distribution and imbalance analysis |
| GET | /api/v1/data/process/stats/{file_id} | Column statistics |
| GET | /api/v1/data/process/distribution/{file_id} | Feature distribution profiles and outlier statistics |
| GET | /api/v1/data/process/correlation/{file_id} | Pairwise numeric correlation matrix |
| GET | /api/v1/data/process/data_metrics/{file_id} | Data metrics and summary |
| POST | /api/v1/data/process/target | Set target column |
| GET | /api/v1/data/process/target | List all target assignments (paginated) |
| GET | /api/v1/data/process/target/{file_id} | Get target for a file |
| DELETE | /api/v1/data/process/target/{file_id} | Delete target assignment |
| POST | /api/v1/data/process/preprocess/{file_id} | Apply transformations |
| POST | /api/v1/data/process/revert/{file_id} | Revert dataset to pre-processed state |
| GET | /api/v1/data/process/backup/{file_id} | Check preprocessing backup status |

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
| GET | /api/v1/model/check-name | Check model name availability |
| GET | /api/v1/model/model-list | List models (paginated) |
| GET | /api/v1/model/training-history | List models with training history metadata |
| POST | /api/v1/model/validate | Validate model graph |
| POST | /api/v1/model/save | Save model architecture |
| PATCH | /api/v1/model/training-config | Update training config |
| GET | /api/v1/model/{model_name}/graph | Get model graph |
| GET | /api/v1/model/{model_id}/profile | Profile model complexity and memory footprint |
| GET | /api/v1/model/{model_id}/architecture | Get layer architecture with shapes and parameters |
| DELETE | /api/v1/model/{model_id} | Delete model |
| POST | /api/v1/model/code | Generate training code |

### Training Jobs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/model/run | Start training job in background (returns 202 Accepted) |
| GET | /api/v1/model/training-jobs | List training jobs for a model |
| GET | /api/v1/model/training-job/{job_id} | Get job status, hyperparams, and latest metrics |
| GET | /api/v1/model/training-job/{job_id}/metrics | Full per-epoch metric history |
| GET | /api/v1/model/training-job/{job_id}/reproducibility | Reproducibility metadata and snapshot details |
| DELETE | /api/v1/model/training-job/{job_id} | Request job cancellation (204 No Content) |
| GET | /api/v1/model/compare | Compare training metrics across multiple jobs (max 5) |

### Hyperparameter Tuning
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/model/tuning/{model_name} | Start hyperparameter tuning session (returns 202 Accepted) |
| GET | /api/v1/model/tuning/{session_id} | Get tuning session details, trial progress, and best metrics |
| DELETE | /api/v1/model/tuning/{session_id} | Cancel tuning session (204 No Content) |
| POST | /api/v1/model/tuning/{session_id}/apply-best | Apply best hyperparameters to model configuration |

### Model Analysis & Interpretability
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/model/analysis/{job_id}/confusion-matrix | Normalized confusion matrix |
| GET | /api/v1/model/analysis/{job_id}/roc-curve | ROC curve coordinates and AUC |
| GET | /api/v1/model/analysis/{job_id}/calibration-curve | Probability calibration and reliability curve |
| GET | /api/v1/model/analysis/{job_id}/regression-metrics | Regression metrics (RMSE, R2, MAE) |
| GET | /api/v1/model/analysis/{job_id}/feature-importance | Permutation feature importance ranking |
| GET | /api/v1/model/analysis/{job_id}/predictions | Sample predictions vs ground truth |
| GET | /api/v1/model/analysis/{job_id}/export | Export analysis results as CSV or JSON |

### Model Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/model/export/{job_id}/formats | List available export formats (SavedModel, TFLite, ONNX, Keras) |
| GET | /api/v1/model/export/{job_id} | Download exported model archive |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Liveness probe |
| GET | /api/v1/ready | Readiness probe with database connectivity check |

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

- Room `{job_id}`: `result ::: <data>` - Streams training events for a job:
  - `{"type": "metric", "job_id": "...", "epoch": 1, ...}` - Per-epoch training and validation metrics
  - `{"type": "status", "job_id": "...", "status": "running" | "completed" | "failed" | "cancelled"}` - Job state transitions
- Room `tuning:{session_id}`: `result ::: <data>` - Streams hyperparameter tuning session updates:
  - `{"type": "tuning_progress", "session_id": "...", "trial": 1, "total_trials": 10, ...}` - Per-trial results
  - `{"type": "tuning_status", "session_id": "...", "status": "completed" | "cancelled" | "failed"}` - Session status

---

*Generated from FastAPI routes*
