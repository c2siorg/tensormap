FROM python:3.9-slim-buster

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false

RUN poetry install --no-dev --no-interaction --no-ansi

COPY . /app

EXPOSE 80

ENV FLASK_APP=app.py

RUN chmod +x ./entrypoint.sh

CMD ["./entrypoint.sh", "flask", "run", "--host=0.0.0.0", "--port=80"]