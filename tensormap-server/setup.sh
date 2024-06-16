command -v poetry >/dev/null 2>&1 || { echo >&2 "Installing Poetry."; curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python; }

poetry install

mkdir -p data