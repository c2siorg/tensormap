FROM python:3.9-slim-buster

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi

COPY . /app

# # The key change: Use RUN to set permissions *after* copying
RUN chmod +x /app/entrypoint.sh

EXPOSE 80

ENV FLASK_APP=app.py

# No need for chmod here anymore
CMD ["/app/entrypoint.sh", "flask", "run", "--host=0.0.0.0", "--port=80"]