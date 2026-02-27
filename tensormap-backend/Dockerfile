FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install poetry

RUN poetry config virtualenvs.in-project true

ENV POETRY_REQUESTS_TIMEOUT=120
ENV POETRY_RETRIES=5

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root --no-interaction --no-ansi

# --- Stage 2: Production Image ---
FROM python:3.12-slim

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the generated virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv
COPY . .

RUN chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 4300

USER appuser

CMD ["uvicorn", "app.main:socket_app", "--host", "0.0.0.0", "--port", "4300"]