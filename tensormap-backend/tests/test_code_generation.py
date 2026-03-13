from unittest.mock import MagicMock

import pytest

from app.services.code_generation import _map_template, generate_code


def test_generate_code_raises_when_model_missing():
    db = MagicMock()
    db.exec.return_value.first.return_value = None

    with pytest.raises(ValueError, match="not found in database"):
        generate_code("missing_model", db)


def test_generate_code_raises_when_file_missing():
    model = MagicMock()
    model.file_id = "missing-file-id"
    model.model_type = 1
    model.target_field = "target"
    model.training_split = 80
    model.optimizer = "adam"
    model.metric = "accuracy"
    model.epochs = 1

    db = MagicMock()
    first_query = MagicMock()
    first_query.first.return_value = model
    second_query = MagicMock()
    second_query.first.return_value = None
    db.exec.side_effect = [first_query, second_query]

    with pytest.raises(ValueError, match="File with id"):
        generate_code("valid_model", db)


def test_map_template_raises_for_unknown_problem_type():
    with pytest.raises(ValueError, match="Unsupported problem type id"):
        _map_template(999)
