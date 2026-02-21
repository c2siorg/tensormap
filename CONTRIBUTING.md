# Contributing to TensorMap

Thank you for your interest in contributing to TensorMap! This guide will help you get started.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/tensormap.git
   cd tensormap
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Backend

```bash
cd tensormap-backend
uv sync                                              # Install dependencies
cp .env.example .env                                 # Configure your PostgreSQL credentials
uv run alembic upgrade head                          # Apply database migrations
uv run uvicorn app.main:socket_app --reload --port 4300  # Start dev server
```

You will need a running PostgreSQL instance. Edit `.env` with your database URL, secret key, and other settings. The default database name is `c2si_db`.

### Frontend

```bash
cd tensormap-frontend
npm install
cp .env.example .env          # Configure API URL if needed
npm start                     # Starts on port 3300
```

### Docker

To run the full stack (PostgreSQL + backend + frontend) with Docker Compose:

```bash
cp .env.example .env                         # Configure PostgreSQL credentials
cp tensormap-backend/.env.example tensormap-backend/.env  # Configure backend settings
docker compose up                            # Start all services
```

This starts PostgreSQL on port 5432, the backend on port 4300, and the frontend on port 3300.

## Development Workflow

1. Make your changes on a feature branch.
2. **Test** your changes:
   - Backend: `cd tensormap-backend && uv run pytest -v`
3. **Lint** your code:
   - Backend: `cd tensormap-backend && uv run ruff check . && uv run ruff format --check .`
   - Frontend: `cd tensormap-frontend && npm run lint`
4. **Format** your code:
   - Backend: `cd tensormap-backend && uv run ruff format .`
   - Frontend: `cd tensormap-frontend && npm run prettier`
5. Commit your changes with a clear, descriptive message.
6. Push your branch and open a Pull Request against `main`.

## Code Style

### Python (Backend)

- **Ruff** for linting and formatting (configured in `tensormap-backend/pyproject.toml` under `[tool.ruff]`).
- Lint rules: E, F, I (import sorting), UP, B, SIM. Max line length of 120 characters. Target Python 3.12.
- Double quotes preferred for strings.
- Pre-commit hooks run `ruff check` and `ruff format` automatically.

### JavaScript (Frontend)

- **ESLint** with React and Hooks plugins (configured in `tensormap-frontend/.eslintrc.cjs`).
- **Prettier** for formatting (semicolons, double quotes, trailing commas, 100 character width). Run with `npm run prettier`.
- PropTypes for component prop validation.

## Reporting Bugs / Requesting Features

Please use the GitHub issue templates when reporting bugs or requesting features:

- **Bug Report**: Use the "Bug Report" template and provide steps to reproduce, expected behavior, and environment details.
- **Feature Request**: Use the "Feature Request" template and describe the problem, proposed solution, and alternatives considered.

## License

By contributing to TensorMap, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
