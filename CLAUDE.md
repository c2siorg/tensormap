# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorMap is a web application for visually creating machine learning models via drag-and-drop (ReactFlow), with reverse engineering to TensorFlow Python code. Users upload datasets, preprocess data, design neural networks visually, train models with real-time progress via WebSocket, and download generated Python training scripts.

## Repository Structure

- **`tensormap-frontend/`** — React 18 + Vite SPA (JavaScript, JSX)
- **`tensormap-backend/`** — Python FastAPI server with TensorFlow/Keras

## Common Commands

### Backend (from `tensormap-backend/`)

```bash
uv sync                                              # Install dependencies
uv run uvicorn app.main:socket_app --reload --port 4300  # Run dev server
uv run pytest -v                                     # Run all tests
uv run pytest tests/unit/test_DataUpload.py -v       # Run single test file
uv run pytest -k "test_get_all_files" -v             # Run test by name
uv run ruff check .                                  # Lint
uv run ruff format --check .                         # Check formatting
uv run ruff check --fix .                            # Auto-fix lint issues
uv run ruff format .                                 # Auto-format
uv run alembic upgrade head                          # Run DB migrations
uv run alembic revision --autogenerate -m "desc"     # Generate new migration
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
```

### Docker (from repo root)

```bash
docker compose up              # Run full stack (DB + backend + frontend)
```

## Ports

| Service    | Port |
|------------|------|
| Frontend   | 3300 |
| Backend    | 4300 |
| PostgreSQL | 5432 |

## Architecture

### Backend

**Framework:** FastAPI wrapped with Socket.IO (`socket_app` in `app/main.py`).

**Layered structure:**
- `app/routers/` — API route definitions (5 routers under `/api/v1`)
- `app/services/` — Business logic (called by routers)
- `app/models/` — SQLModel database models
- `app/shared/` — Error mappings, logging config

**API routes:**
- `data_upload` — File upload/list/delete (CSV and ZIP image datasets)
- `data_process` — Target field, metrics, preprocessing, image properties
- `deep_learning` — Model validate, code generation, model training, model list
- `project` — Project CRUD (create, read, update, delete)
- `health` — Health check endpoint

**Key data flow:** ReactFlow graph JSON → `model_generation()` converts to Keras JSON → `tf.keras.models.model_from_json()` validates → training via `model_run()` with `CustomProgressBar` Keras callback emitting Socket.IO events.

**WebSocket namespace:** `/dl-result` — emits `"result :::"` events with training progress (epoch, batch, evaluation results).

**Code generation:** Jinja2 templates in `app/templates/` render downloadable Python training scripts.

**Database:** PostgreSQL via SQLModel/SQLAlchemy. Alembic migrations in `migrations/versions/`. Migrations auto-run on app startup.

**Config:** `app/config.py` uses pydantic-settings, reads from environment variables (see `.env.example`).

**All API responses** follow the format: `{"success": bool, "message": str, "data": ...}`.

### Frontend

**Stack:** React 18 + Vite + Recoil + shadcn/ui + Tailwind CSS + ReactFlow.

**Page routing** (React Router v6 in `src/App.jsx`):
- `/projects` — Project listing and creation
- `/workspace/:projectId/dataset` — Upload CSV/ZIP files (canonical; `/datasets` redirects here)
- `/workspace/:projectId/process` — Data visualization, correlation matrix, preprocessing
- `/workspace/:projectId/models` — Neural network builder canvas
- `/workspace/:projectId/training` — Model training with real-time progress

**Component organization:**
- `src/containers/` — Page-level components (ProjectsPage, DataUpload, DataProcess, DeepLearning, Training)
- `src/components/` — Reusable UI (AppTopBar, Workspace, DragAndDropCanvas, Process, Upload, PropertiesBar)
- `src/hooks/` — Custom React hooks (useProjectData)
- `src/services/` — Axios API calls (FileServices, ModelServices, ProjectServices)
- `src/shared/` — Recoil atoms, Axios instance
- `src/constants/` — URL and string constants

**Drag-and-drop (ReactFlow):** Custom node types (`InputNode`, `DenseNode`, `ConvNode`, `FlattenNode`, `DropoutNode`, `MaxPoolingNode`, `GlobalAvgPoolNode`) in `src/components/DragAndDropCanvas/CustomNodes/`. Sidebar provides draggable node palette. `Helpers.jsx` handles graph validation (BFS connectivity check) and model JSON generation.

**State management:** Recoil with `models`, `currentProject`, and `projectFiles` atoms.

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
