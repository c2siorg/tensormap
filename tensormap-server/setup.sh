command -v poetry >/dev/null 2>&1 || { echo >&2 "Installing Poetry.";curl -sSL https://install.python-poetry.org | python3 - ;}

poetry install

mkdir -p data