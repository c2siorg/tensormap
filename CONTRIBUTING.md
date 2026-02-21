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
poetry install
poetry shell
cp .env.example .env          # Configure your PostgreSQL credentials
flask db upgrade              # Apply database migrations
python app.py                 # Starts on port 4300
```

You will need a running PostgreSQL instance. Edit `.env` with your database host, name, user, and password. The default database name is `c2si_db`.

### Frontend

```bash
cd tensormap-frontend
npm install
cp .env.example .env          # Configure API URL if needed
npm start                     # Starts on port 3300
```

### Docker

If you prefer Docker, you can build and run each service independently:

**Backend:**
```bash
cd tensormap-backend
docker build -t tensormap-backend .
docker run -p 4300:80 --env-file .env tensormap-backend
```

**Frontend:**
```bash
cd tensormap-frontend
docker build -t tensormap-frontend .
docker run -p 3300:3300 tensormap-frontend
```

## Development Workflow

1. Make your changes on a feature branch.
2. **Test** your changes:
   - Backend: `cd tensormap-backend && pytest`
   - Frontend: `cd tensormap-frontend && npm test`
3. **Lint** your code:
   - Backend: `cd tensormap-backend && flake8 .`
   - Frontend: `cd tensormap-frontend && npm run lint`
4. Commit your changes with a clear, descriptive message.
5. Push your branch and open a Pull Request against `main`.

## Code Style

### Python (Backend)

- **Flake8** with a max line length of 120 characters (configured in `tensormap-backend/.flake8`).
- **isort** for consistent import ordering.
- Double quotes preferred for strings.
- Type hints on all public service functions.

### JavaScript (Frontend)

- **ESLint** with React and Hooks plugins (configured in `tensormap-frontend/.eslintrc.cjs`).
- **Prettier** for formatting (semicolons, double quotes, trailing commas, 100 character width).
- PropTypes for component prop validation.

## Reporting Bugs / Requesting Features

Please use the GitHub issue templates when reporting bugs or requesting features:

- **Bug Report**: Use the "Bug Report" template and provide steps to reproduce, expected behavior, and environment details.
- **Feature Request**: Use the "Feature Request" template and describe the problem, proposed solution, and alternatives considered.

## License

By contributing to TensorMap, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
