# TensorMap

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build Status](https://github.com/c2siorg/tensormap/actions/workflows/test.yml/badge.svg)](https://github.com/c2siorg/tensormap/actions/workflows/test.yml)

A web application for visually creating machine learning algorithms via drag-and-drop, with reverse engineering to TensorFlow code.

## Features

- Drag-and-drop neural network design using ReactFlow
- Reverse engineer visual models to TensorFlow Python code
- Real-time model training with live progress via WebSocket
- CSV and image dataset upload and preprocessing
- Correlation matrix visualization and target field selection

## Prerequisites

- [Node.js](https://nodejs.org/) >= 18
- Python 3.12+
- PostgreSQL
- [Docker](https://docs.docker.com/get-docker/) (optional, for containerized setup)

## Getting Started

### Quick Start (Docker)

```bash
# Backend
cd tensormap-backend
docker build -t tensormap-backend .
docker run -p 4300:4300 --env-file .env tensormap-backend

# Frontend
cd tensormap-frontend
docker build -t tensormap-frontend .
docker run -p 3300:3300 tensormap-frontend
```

### Backend

```bash
cd tensormap-backend
cp .env.example .env          # Configure DB credentials
uv sync
uv run uvicorn app.main:socket_app --reload --port 4300
```

### Frontend

```bash
cd tensormap-frontend
cp .env.example .env          # Optional: configure API URL
npm install
npm start
```

| Service  | Port |
|----------|------|
| Frontend | 3300 |
| Backend  | 4300 |

## Branch Protection Rules

To enforce code quality, ensure that you set up GitHub Branch Protection rules for the `main` branch. 
1. Go to **Settings > Branches**.
2. Add branch protection rule for `main`.
3. Check **"Require status checks to pass before merging"**.
4. Search and select `backend-tests` and `frontend-tests`.

## Development Setup

### Running Backend Tests
To run the backend tests locally with SQLite (fallback):
```bash
cd tensormap-backend && pytest tests/ -v
```
To run tests with PostgreSQL:
```bash
export DATABASE_URL=postgresql://test:test@localhost:5432/tensormap_test
cd tensormap-backend && pytest tests/ -v
```

### Running Frontend Tests
To run frontend tests locally:
```bash
cd tensormap-frontend && npx vitest
```

## Project Structure

```
tensormap/
  tensormap-backend/    # Python FastAPI server
  tensormap-frontend/   # React + Vite SPA
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache 2.0](LICENSE)

## Author

[Oshan Mudannayake](mailto:oshan.ivantha@gmail.com)

For questions or queries about this project, please reach out via email.
