# TensorMap

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build Status](https://github.com/c2siorg/tensormap/actions/workflows/test.yml/badge.svg)](https://github.com/c2siorg/tensormap/actions/workflows/test.yml)

A web application for visually creating machine learning algorithms via drag-and-drop, with reverse engineering to TensorFlow code.

## Features

- Drag-and-drop neural network design using ReactFlow
- Reverse engineer visual models to TensorFlow Python code
- Real-time model training with live progress via WebSocket
- Automated hyperparameter tuning (Grid and Random search) with real-time trial progression
- Post-training model evaluation and interpretability (confusion matrix, ROC curves, feature importance, probability calibration, and exportable reports)
- Multi-format model export (SavedModel, TFLite, ONNX, and Keras)
- CSV and image dataset upload, validation, and preprocessing with undo/revert support
- Dataset quality profiler and target class imbalance analyzer
- Multi-run training job comparison charts
- Live model check: inferred output shapes and parameter counts on every layer, with problems pointed at the layer that causes them (see [graph analysis](tensormap-backend/docs/GRAPH_ANALYSIS.md))

## Prerequisites

- [Node.js](https://nodejs.org/) >= 22.12.0
- Python 3.12+
- PostgreSQL
- [Docker](https://docs.docker.com/get-docker/) (optional, for containerized setup)

## Getting Started

### Quick start with Docker Compose

```bash
git clone https://github.com/c2siorg/tensormap.git
cd tensormap
cp .env.example .env
cp tensormap-backend/.env.example tensormap-backend/.env
docker compose up --build
```

There are two environment files. The one at the repository root holds the PostgreSQL credentials used by the `db` service. The one in `tensormap-backend/` holds the application settings; Compose overrides its `DATABASE_URL` so the backend connects to the `db` container.

| Service    | URL                           |
|------------|-------------------------------|
| Frontend   | http://localhost:3300         |
| Backend    | http://localhost:4300         |
| API docs   | http://localhost:4300/docs    |
| PostgreSQL | localhost:5432                |

Uploaded datasets are written to `tensormap-backend/data/` on the host, and database contents are kept in the `pgdata` volume. The first build downloads TensorFlow and takes a while. If port 5432 is already used by a local PostgreSQL, stop that server or change the left-hand number of the `db` port mapping in `docker-compose.yml`.

### Backend

```bash
cd tensormap-backend
cp .env.example .env          # Configure DB credentials
uv sync --extra dev           # --extra dev adds pytest and ruff
uv run uvicorn app.main:socket_app --reload --port 4300
```

Database migrations are applied when the server starts. `.env.example` points at a local PostgreSQL instance; to use SQLite instead, set `DATABASE_URL=sqlite:///./tensormap.db`. The server must be started as `app.main:socket_app`, because the Socket.IO server wraps the FastAPI app. See the [backend README](tensormap-backend/README.md) for configuration and architecture.

### Frontend

```bash
cd tensormap-frontend
cp .env.example .env          # Optional: configure API URL
npm install
npm start
```

The frontend reads the backend addresses from `VITE_API_BASE_URL` and `VITE_WS_URL`. See the [frontend README](tensormap-frontend/README.md).

## Branch Protection Rules

To enforce code quality, ensure that you set up GitHub Branch Protection rules for the `main` branch. 
1. Go to **Settings > Branches**.
2. Add branch protection rule for `main`.
3. Check **"Require status checks to pass before merging"**.
4. Search and select `backend-tests` and `frontend-tests`.

## Development Setup

| Check             | Run from              | Command                                                                |
|-------------------|-----------------------|------------------------------------------------------------------------|
| Backend tests     | `tensormap-backend/`  | `uv run pytest -v`                                                     |
| Integration tests | `tensormap-backend/`  | `uv run pytest ../tests/integration/ -v`                               |
| Backend lint      | `tensormap-backend/`  | `uv run ruff check .` and `uv run ruff format --check .`               |
| Frontend tests    | `tensormap-frontend/` | `npm test`                                                             |
| Frontend lint     | `tensormap-frontend/` | `npm run lint` and `npx prettier --check "src/**/*.{js,jsx,json,css}"` |

The backend tests use SQLite unless `DATABASE_URL` is set. To run them against PostgreSQL:

```bash
export DATABASE_URL=postgresql://test:test@localhost:5432/tensormap_test
cd tensormap-backend && uv run pytest -v
```

## Project Structure

```
tensormap/
├── tensormap-backend/    FastAPI and Socket.IO server, TensorFlow training, Alembic migrations
├── tensormap-frontend/   React and Vite single-page app
├── tests/                Integration tests that run against the backend
└── docker-compose.yml    PostgreSQL, backend and frontend
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache 2.0](LICENSE)

## Author

[Oshan Mudannayake](mailto:oshan.ivantha@gmail.com)

For questions or queries about this project, please reach out via email.
