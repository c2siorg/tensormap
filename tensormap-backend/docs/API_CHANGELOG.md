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
| GET | /api/v1/data/process/stats/{file_id} | Column statistics |
| GET | /api/v1/data/process/correlation/{file_id} | Correlation matrix |
| GET | /api/v1/data/process/data_metrics/{file_id} | Data metrics |
| POST | /api/v1/data/process/target | Set target column |
| GET | /api/v1/data/process/target | List all target assignments |
| GET | /api/v1/data/process/target/{file_id} | Get target for a file |
| DELETE | /api/v1/data/process/target/{file_id} | Delete target assignment |
| POST | /api/v1/data/process/preprocess/{file_id} | Apply transformations |

### Deep Learning
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/deep_learning/model/model-list | List models |
| POST | /api/v1/deep_learning/model/validate | Validate model graph |
| POST | /api/v1/deep_learning/model/save | Save model architecture |
| PATCH | /api/v1/deep_learning/model/training-config | Update training config |
| GET | /api/v1/deep_learning/model/{model_name}/graph | Get model graph |
| DELETE | /api/v1/deep_learning/model/{model_id} | Delete model |
| POST | /api/v1/deep_learning/model/code | Generate training code |
| POST | /api/v1/deep_learning/model/run | Run model training |

### Project
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/project | List projects |
| POST | /api/v1/project | Create project |
| GET | /api/v1/project/{project_id} | Get project details |
| PATCH | /api/v1/project/{project_id} | Update project |
| DELETE | /api/v1/project/{project_id} | Delete project |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Liveness probe |
| GET | /ready | Readiness probe |

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
- 422: Validation Error
- 429: Rate Limited
- 500: Server Error

## Headers

- X-Request-ID: Request tracing ID
- X-RateLimit-Remaining: Requests left

## Pagination

Use offset and limit query params. Max limit: 500.

## WebSocket Events

- training:start
- training:progress
- training:complete
- training:error
