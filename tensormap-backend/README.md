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
   CORS_ALLOWED_ORIGIN=http://localhost:3300
   ```
4. Run database migrations:
   ```bash
   uv run alembic upgrade head
   ```
5. Start the server:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 4300 --reload
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
│   │   └── ml.py             # ModelBasic, ModelConfigs, ModelResults
│   ├── schemas/
│   │   ├── data_process.py   # Request validation schemas
│   │   └── deep_learning.py
│   ├── routers/
│   │   ├── data_upload.py    # /data/upload endpoints
│   │   ├── data_process.py   # /data/process endpoints
│   │   └── deep_learning.py  # /model endpoints
│   ├── services/
│   │   ├── data_upload.py    # File upload business logic
│   │   ├── data_process.py   # Data processing business logic
│   │   ├── deep_learning.py  # Model validation/execution logic
│   │   ├── model_generation.py  # ReactFlow graph to Keras JSON
│   │   ├── code_generation.py   # Jinja2 TF code generation
│   │   └── model_run.py      # Model training execution
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
