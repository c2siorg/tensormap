# TensorMap

[![CI](https://github.com/c2siorg/tensormap/actions/workflows/ci.yml/badge.svg)](https://github.com/c2siorg/tensormap/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

## Running Tests

```bash
# Backend
cd tensormap-backend && uv run pytest

# Frontend
cd tensormap-frontend && npm test
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
