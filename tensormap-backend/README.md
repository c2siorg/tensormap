## Backend setup and architecture

First, make sure you have PostgreSQL server and Python 3.12+ installed in your system.

### Starting the Server using Docker

Requirements: docker

```bash
docker build -t tensormap-backend .
docker run -p 4300:4300 --env-file .env tensormap-backend
```

### Starting PostgreSQL using Docker

```bash
docker run -d \
  --name database \
  -e POSTGRES_DB=c2si_db \
  -e POSTGRES_USER=c2si \
  -e POSTGRES_PASSWORD=c2si \
  -p 5432:5432 \
  postgres:16
```

### Installation instructions

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Copy `.env.example` to `.env` and configure:
   ```
   DATABASE_URL=postgresql+psycopg2://c2si:c2si@localhost:5432/c2si_db
   SECRET_KEY=changeme
   CORS_ALLOWED_ORIGINS=http://localhost:3300
   ```
4. Run database migrations:
   ```bash
   uv run alembic upgrade head
   ```
5. Start the server:
   ```bash
   uv run uvicorn app.main:socket_app --host 0.0.0.0 --port 4300 --reload
   ```

### Application Architecture

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py             # Pydantic Settings configuration
│   ├── database.py           # SQLModel engine and session
│   ├── exceptions.py         # Standardized error handling
│   ├── socketio_instance.py  # Async Socket.IO server
│   ├── models/
│   │   ├── data.py           # DataFile, DataProcess, ImageProperties
│   │   ├── ml.py             # ModelBasic, ModelConfigs
│   │   ├── project.py        # Project
│   │   ├── training_job.py   # TrainingJob, TrainingStatus
│   │   ├── training_metric.py# TrainingMetric
│   │   └── tuning_session.py # TuningSession, TuningStatus, TuningStrategy
│   ├── schemas/
│   │   ├── data_process.py   # Request validation schemas for data processing
│   │   ├── deep_learning.py  # Model and training config schemas
│   │   ├── project.py        # Project schemas
│   │   ├── training.py       # Training job schemas
│   │   └── tuning.py         # Hyperparameter tuning schemas
│   ├── routers/
│   │   ├── analysis.py       # /model/analysis post-training interpretability endpoints
│   │   ├── data_process.py   # /data/process endpoints
│   │   ├── data_upload.py    # /data/upload endpoints
│   │   ├── deep_learning.py  # /model architecture and code gen endpoints
│   │   ├── export.py         # /model/export format and download endpoints
│   │   ├── health.py         # /health and /ready probe endpoints
│   │   ├── layers.py         # /layers registry and graph validation/analysis
│   │   ├── project.py        # /project CRUD endpoints
│   │   ├── training.py       # /model training jobs, metrics, and comparison endpoints
│   │   └── tuning.py         # /model/tuning hyperparameter tuning endpoints
│   ├── services/
│   │   ├── code_generation.py   # Jinja2 TF code generation
│   │   ├── data_process.py   # Data processing and transformation logic
│   │   ├── data_upload.py    # File upload and extraction business logic
│   │   ├── dataset_quality.py# Dataset quality profiling and validation
│   │   ├── deep_learning.py  # Model validation and configuration logic
│   │   ├── interpretability.py # Confusion matrix, ROC, feature importance
│   │   ├── model_export.py   # SavedModel, TFLite, ONNX, and Keras exports
│   │   ├── model_generation.py # ReactFlow graph to Keras JSON
│   │   ├── model_profiler.py # Layer complexity and parameter profiling
│   │   ├── model_run.py      # Background model training execution
│   │   ├── project.py        # Project management business logic
│   │   ├── reproducibility.py# Environment and data snapshot capture
│   │   ├── training_service.py # Training job orchestration and metrics persistence
│   │   └── tuning_service.py # Grid/random search hyperparameter tuning
│   ├── callbacks/            # Keras training callbacks (metrics, cancellation, checkpoint)
│   ├── ir/                   # Intermediate Representation schema, translator, and analysis
│   ├── layers/               # Dynamic layer registry and specifications
│   └── shared/
│       ├── constants.py
│       ├── enums.py
│       ├── errors.py
│       └── logging_config.py
├── migrations/               # Alembic migrations
├── templates/                # Jinja2 code generation templates
├── tests/
├── alembic.ini
├── pyproject.toml
└── Dockerfile
```

### Database modifications

Once you change database models, generate a new migration:

```bash
uv run alembic revision --autogenerate -m "description"
```

To apply migrations:

```bash
uv run alembic upgrade head
```

Migrations are also auto-applied on app startup via the lifespan handler.

## Testing

For the backend, `pytest` is used for testing. To run:

```bash
uv run pytest -v
```

## Linting

```bash
uv run ruff check .
uv run ruff format --check .
```
