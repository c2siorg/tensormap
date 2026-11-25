# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorMap is a web application for visually creating machine learning models via drag-and-drop (ReactFlow), with reverse engineering to TensorFlow Python code. Users upload datasets, preprocess data, design neural networks visually, train models with real-time progress via WebSocket, and download generated Python training scripts.

## Repository Structure

- **`tensormap-frontend/`** — React 18 + Vite SPA (JavaScript/JSX, with newer code in TypeScript/TSX)
- **`tensormap-backend/`** — Python FastAPI server with TensorFlow/Keras
- **`tests/`** — Integration tests that run against the backend (run from `tensormap-backend/`)

## Common Commands

### Backend (from `tensormap-backend/`)

Run backend commands from this directory: `./data`, `./templates`, `./exports` and `alembic.ini` are resolved relative to the working directory.

```bash
uv sync --extra dev                                  # Install dependencies
uv run uvicorn app.main:socket_app --reload --port 4300  # Run dev server
uv run pytest -v                                     # Run all tests
uv run pytest -m "not slow" -v                       # Skip tests that train a real model
uv run pytest tests/test_project_service.py -v       # Run single test file
uv run pytest -k "test_deletes_db_and_removes_files" -v    # Run test by name
uv run pytest ../tests/integration/ -v               # Run the integration tests in ../tests
uv run ruff check .                                  # Lint
uv run ruff format --check .                         # Check formatting
uv run ruff check --fix .                            # Auto-fix lint issues
uv run ruff format .                                 # Auto-format
uv run alembic upgrade head                          # Run DB migrations
uv run alembic revision --autogenerate -m "desc"     # Generate new migration
uv run alembic heads                                 # Must list exactly one head
```

### Frontend (from `tensormap-frontend/`)

```bash
npm install                    # Install dependencies
npm start                      # Dev server (port 3300)
npm test                       # Run Vitest tests
npm run build                  # Production build
npm run lint                   # ESLint check
npm run lint:fix               # ESLint auto-fix
npm run prettier               # Format with Prettier
npx prettier --check "src/**/*.{js,jsx,json,css}"   # Prettier check, as run in CI
```

### Docker (from repo root)

```bash
cp .env.example .env                    # PostgreSQL credentials for the db service
cp tensormap-backend/.env.example tensormap-backend/.env
docker compose up --build              # Run full stack (DB + backend + frontend)
```

## Ports

| Service    | Port |
|------------|------|
| Frontend   | 3300 |
| Backend    | 4300 |
| PostgreSQL | 5432 |

The backend serves interactive API docs at `/docs`. `GET /health` and `GET /ready` are not under `/api/v1`.

## Architecture

### Backend

**Framework:** FastAPI wrapped with Socket.IO (`socket_app` in `app/main.py`). Start it as `app.main:socket_app`.

**Layered structure:**
- `app/routers/` — API route definitions (10 routers under `/api/v1`)
- `app/schemas/` — Pydantic request/response models
- `app/services/` — Business logic (called by routers); returns `(body, status_code)` tuples that routers wrap in `JSONResponse`
- `app/models/` — SQLModel database models
- `app/ir/` — Typed intermediate representation (IR) of a model graph (`schema.py`, `translator.py`) and the static analyzer (`analysis/`)
- `app/layers/` — Layer registry (`registry.py`): supported layers and their parameters, also served to the frontend
- `app/generators/` — Builds a Keras model from an IR graph (`tensorflow_generator.py`)
- `app/callbacks/` — Keras callbacks: `MetricsCallback` (persists and emits metrics), `CancellationCheckCallback`
- `app/shared/` — Error mappings, logging config

**API routes:**
- `data_upload` — File upload/list/delete (CSV and ZIP image datasets)
- `data_process` — Target field, metrics, preprocessing, image properties
- `deep_learning` — Model validate, code generation, model training, model list
- `layers` — Layer registry grouped by category, graph validation, graph analysis (`/layers/...`)
- `training` — Start a training job, list/get/cancel jobs, job metrics, compare jobs (`/model/run`, `/model/training-job/...`, `/model/compare`)
- `tuning` — Hyperparameter tuning sessions (`/model/tuning/...`)
- `analysis` — Confusion matrix, ROC curve, feature importance, predictions, residuals (`/model/analysis/{job_id}/...`)
- `export` — Export a trained model as Keras, SavedModel, TFLite or ONNX (`/model/export/{job_id}`)
- `project` — Project CRUD (create, read, update, delete)
- `health` — Health check endpoint

**Key data flow:** ReactFlow graph JSON → `reactflow_to_ir()` (`app/ir/translator.py`) → IR validation → `TensorFlowGenerator` builds the Keras model using the layer registry. If that fails, `model_generation()` in `app/services/model_generation.py` (the earlier converter) is used as a fallback. The Keras JSON is written to `templates/json-model/` and the model is saved to the database. `POST /model/run` creates a `TrainingJob` and runs `model_run()` in a background thread with `MetricsCallback` and `CancellationCheckCallback`. `CustomProgressBar` in `model_run.py` is only used for runs without a job id.

**Graph analysis:** `POST /layers/analyze-graph` computes per-layer output shapes, parameter counts and diagnostics without importing TensorFlow. See `tensormap-backend/docs/GRAPH_ANALYSIS.md`.

**WebSocket namespace:** `/dl-result`. Clients emit `subscribe_job` with `{"job_id": ...}` to join a job's room and receive a catch-up of stored metrics. Events are sent under the name `result :::` with a versioned payload (`version`, `type`, `job_id`, `timestamp`) where `type` is `metrics`, `status` or `catchup`. Tuning uses `subscribe_tuning` with `{"tuning_id": ...}` and `tuning_progress` events.

**Code generation:** Jinja2 templates in `app/templates/` render downloadable Python training scripts.

**Database:** PostgreSQL via SQLModel/SQLAlchemy. Alembic migrations in `migrations/versions/`. Migrations auto-run on app startup.

**Config:** `app/config.py` uses pydantic-settings, reads from environment variables (see `.env.example`). `DATABASE_URL` and `SECRET_KEY` are required.

**All API responses** follow the format: `{"success": bool, "message": str, "data": ...}`.

### Frontend

**Stack:** React 18 + Vite + Recoil + shadcn/ui + Tailwind CSS + ReactFlow + Recharts. Older code is `.js`/`.jsx`; newer code is `.ts`/`.tsx`.

**Page routing** (React Router v6 in `src/App.jsx`):
- `/projects` — Project listing and creation
- `/workspace/:projectId/dataset` — Upload CSV/ZIP files (canonical; `/datasets` redirects here)
- `/workspace/:projectId/process` — Data visualization, correlation matrix, preprocessing
- `/workspace/:projectId/models` — Neural network builder canvas
- `/workspace/:projectId/training` — Model training with real-time progress

**Component organization:**
- `src/containers/` — Page-level components (ProjectsPage, DataUpload, DataProcess, DeepLearning, Training)
- `src/components/` — Reusable UI (AppTopBar, Workspace, DragAndDropCanvas, Process, Upload, PropertiesBar)
- `src/hooks/` — Custom React hooks (`useProjectData`, `useLayerRegistry`, `useGraphAnalysis`, `useUndoRedo`, `useTrainingMetrics`, and others)
- `src/services/` — API and socket clients (`FileServices`, `ModelServices`, `ProjectServices`, `socketService`, plus typed services for tuning, comparison and graph analysis)
- `src/shared/` — Recoil atoms, Axios instance, logger
- `src/types/` — TypeScript types for the registry, graph analysis, training, analysis and export
- `src/constants/` — URL and string constants

**Drag-and-drop (ReactFlow):** `Canvas.jsx` hosts the editor. The layer palette and property panel are built from the backend layer registry (`useLayerRegistry`), and `nodes/GenericLayerNode.tsx` renders any registry layer. The original per-layer node components (`InputNode`, `DenseNode`, `ConvNode`, `FlattenNode`, `DropoutNode`, `MaxPoolingNode`, `GlobalAvgPoolNode`) are in `src/components/DragAndDropCanvas/CustomNodes/`. `Helpers.ts` handles graph validation (BFS connectivity check) and model JSON generation. `useGraphAnalysis` calls the backend analyzer while the graph is edited.

**State management:** Recoil with `models`, `currentProject`, `projectFiles` and `trainingHistory` atoms.

**WebSocket:** Socket.IO client connects to `/dl-result` namespace for real-time training progress.

**API base URL:** Configured via `VITE_API_BASE_URL` env var (default: `http://127.0.0.1:4300/api/v1`).

## Code Style

### Backend (Python)
- **Linter/formatter:** Ruff (line-length 120, target Python 3.12)
- **Rules:** E, F, I (isort), UP, B, SIM
- **Quotes:** Double quotes

### Frontend (JavaScript)
- **Linter:** ESLint with react, react-hooks, prettier plugins
- **Formatter:** Prettier (semicolons, double quotes, trailing commas, 100 char width, 2-space indent)
- **Prop validation:** PropTypes (warn level)
- **Testing:** Vitest + @testing-library/react (jsdom environment)

## Pull Requests

- Branches: `feat/phase-N-description`, `fix/description`, `test/description`
- Commits follow Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`)
- Keep PRs under about 800 lines
- The lint workflow asks for a branch rebased on `main` with no merge commits, squashed into a single commit
- Run Ruff (backend) and ESLint and Prettier (frontend), and the test suites, before pushing
