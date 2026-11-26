"""Tests for multi-dimensional Input layer shapes and ImageProperties API/service."""

import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.database import get_db
from app.ir.analysis import analyze_graph
from app.ir.schema import (
    Conv2DParams,
    DenseParams,
    FlattenParams,
    InputParams,
    IREdge,
    IRGraph,
    IRNode,
    LSTMParams,
    MaxPool2DParams,
    parse_input_shape,
)
from app.ir.translator import ir_to_keras_build_args, reactflow_to_ir
from app.layers.registry import LAYER_REGISTRY, ParamType
from app.main import app
from app.models.data import DataFile, ImageProperties

IN_CI = os.getenv("CI", "").lower() == "true" or os.getenv("GITHUB_ACTIONS", "").lower() == "true"

client = TestClient(app)


@pytest.fixture(name="db_session")
def fixture_db_session():
    """In-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class TestMultiDimInputShape:
    """Verify multi-dimensional InputParams parsing, translation, shape analysis, and model building."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (784, 784),
            ("784", 784),
            (" 28, 28, 3 ", [28, 28, 3]),
            ("(224, 224, 3)", [224, 224, 3]),
            ("[50, 16]", [50, 16]),
            ([28, 28, 1], [28, 28, 1]),
            ((64, 64, 3), [64, 64, 3]),
            ([128], 128),
            ("(784,)", 784),
        ],
    )
    def test_parse_input_shape_valid_formats(self, raw, expected):
        assert parse_input_shape(raw) == expected
        params = InputParams(layer_type="input", shape=raw)
        assert params.shape == expected

    @pytest.mark.parametrize(
        "bad",
        [
            True,
            False,
            0,
            -5,
            "",
            "   ",
            "()",
            [],
            [28, 0, 3],
            [28, -1, 3],
            [28, True],
            "28,,3",
            "28, 0, 3",
            "abc",
            "28.5, 28, 3",
        ],
    )
    def test_input_params_rejects_invalid_shapes(self, bad):
        with pytest.raises(ValidationError):
            InputParams(layer_type="input", shape=bad)

    def test_registry_input_shape_accepts_strings_and_preserves_min(self):
        spec = LAYER_REGISTRY["input"].params["shape"]
        assert spec.type == ParamType.STRING
        assert spec.min == 1
        assert spec.required is True

    def test_ir_round_trip_multidim_input(self):
        graph = IRGraph(
            nodes=[
                IRNode(id="in1", node_params=InputParams(layer_type="input", shape="28, 28, 3")),
                IRNode(id="c1", node_params=Conv2DParams(layer_type="conv2d", filters=16, kernel_size=3)),
            ],
            edges=[IREdge(id="e1", source_id="in1", target_id="c1")],
        )
        dumped = graph.model_dump()
        restored = IRGraph.model_validate(dumped)
        assert restored.nodes[0].node_params.shape == [28, 28, 3]  # type: ignore[union-attr]

    def test_translator_preserves_legacy_3d_dims_and_registry_shape(self):
        rf_legacy = {
            "nodes": [
                {
                    "id": "in-legacy",
                    "type": "custominput",
                    "data": {"params": {"dim-1": 64, "dim-2": 64, "dim-3": 3}},
                },
                {
                    "id": "in-reg",
                    "type": "input",
                    "data": {"params": {"shape": "224, 224, 3"}},
                },
            ],
            "edges": [],
        }
        graph = reactflow_to_ir(rf_legacy)
        assert graph.nodes[0].node_params.shape == [64, 64, 3]  # type: ignore[union-attr]
        assert graph.nodes[1].node_params.shape == [224, 224, 3]  # type: ignore[union-attr]

        assert ir_to_keras_build_args(graph.nodes[0])["shape"] == (64, 64, 3)
        assert ir_to_keras_build_args(graph.nodes[1])["shape"] == (224, 224, 3)

    def test_shape_analysis_cnn_direct_from_3d_input(self):
        graph = IRGraph(
            nodes=[
                IRNode(id="in", node_params=InputParams(layer_type="input", shape=[28, 28, 3])),
                IRNode(
                    id="conv",
                    node_params=Conv2DParams(layer_type="conv2d", filters=16, kernel_size=3, padding="same"),
                ),
                IRNode(id="pool", node_params=MaxPool2DParams(layer_type="maxpool2d", pool_size=2)),
                IRNode(id="flat", node_params=FlattenParams(layer_type="flatten")),
                IRNode(id="out", node_params=DenseParams(layer_type="dense", units=10, activation="softmax")),
            ],
            edges=[
                IREdge(id="e1", source_id="in", target_id="conv"),
                IREdge(id="e2", source_id="conv", target_id="pool"),
                IREdge(id="e3", source_id="pool", target_id="flat"),
                IREdge(id="e4", source_id="flat", target_id="out"),
            ],
        )
        result = analyze_graph(graph)
        assert result.complete is True
        assert result.diagnostics == []
        by_id = {n.node_id: n.output_shape for n in result.nodes}
        assert by_id["in"] == (28, 28, 3)
        assert by_id["conv"] == (28, 28, 16)
        assert by_id["pool"] == (14, 14, 16)
        assert by_id["flat"] == (3136,)
        assert by_id["out"] == (10,)

    def test_shape_analysis_rnn_direct_from_2d_input(self):
        graph = IRGraph(
            nodes=[
                IRNode(id="in", node_params=InputParams(layer_type="input", shape="50, 16")),
                IRNode(id="lstm", node_params=LSTMParams(layer_type="lstm", units=32, return_sequences=False)),
                IRNode(id="out", node_params=DenseParams(layer_type="dense", units=4, activation="softmax")),
            ],
            edges=[
                IREdge(id="e1", source_id="in", target_id="lstm"),
                IREdge(id="e2", source_id="lstm", target_id="out"),
            ],
        )
        result = analyze_graph(graph)
        assert result.complete is True
        assert result.diagnostics == []
        by_id = {n.node_id: n.output_shape for n in result.nodes}
        assert by_id["in"] == (50, 16)
        assert by_id["lstm"] == (32,)
        assert by_id["out"] == (4,)

    def test_tensorflow_generator_builds_direct_3d_cnn(self):
        if IN_CI:
            pytest.skip("TensorFlow model building runs locally")
        from app.generators.tensorflow_generator import TensorFlowGenerator

        graph = IRGraph(
            nodes=[
                IRNode(id="in", node_params=InputParams(layer_type="input", shape=[28, 28, 3])),
                IRNode(id="conv", node_params=Conv2DParams(layer_type="conv2d", filters=8, kernel_size=3)),
                IRNode(id="flat", node_params=FlattenParams(layer_type="flatten")),
                IRNode(id="out", node_params=DenseParams(layer_type="dense", units=3, activation="softmax")),
            ],
            edges=[
                IREdge(id="e1", source_id="in", target_id="conv"),
                IREdge(id="e2", source_id="conv", target_id="flat"),
                IREdge(id="e3", source_id="flat", target_id="out"),
            ],
        )
        model = TensorFlowGenerator().build_model(graph)
        assert model.input_shape == (None, 28, 28, 3)
        assert model.output_shape == (None, 3)


class TestImagePropertiesEndpoints:
    """Verify GET and PATCH /api/v1/data/process/image/{file_id} endpoints."""

    def test_get_image_properties_404_when_file_missing(self, db_session):
        app.dependency_overrides[get_db] = lambda: db_session
        try:
            resp = client.get(f"/api/v1/data/process/image/{uuid4()}")
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db, None)

    def test_get_image_properties_400_for_non_zip_file(self, db_session):
        file_id = uuid4()
        db_session.add(DataFile(id=file_id, file_name="iris.csv", file_type="csv", disk_name="iris.csv"))
        db_session.commit()

        app.dependency_overrides[get_db] = lambda: db_session
        try:
            resp = client.get(f"/api/v1/data/process/image/{file_id}")
            assert resp.status_code == 400
            assert "ZIP" in resp.json()["message"]
        finally:
            app.dependency_overrides.pop(get_db, None)

    def test_get_and_patch_image_properties_success(self, db_session, tmp_path):
        file_id = uuid4()
        db_session.add(DataFile(id=file_id, file_name="animals.zip", file_type="zip", disk_name="animals_123.zip"))
        db_session.add(ImageProperties(id=file_id, image_size=224, batch_size=32, color_mode="rgb", label_mode="int"))
        db_session.commit()

        extracted_dir = tmp_path / "animals_123"
        (extracted_dir / "cats").mkdir(parents=True)
        (extracted_dir / "dogs").mkdir(parents=True)
        (extracted_dir / "cats" / "c1.png").write_bytes(b"png")
        (extracted_dir / "cats" / "c2.jpg").write_bytes(b"jpg")
        (extracted_dir / "dogs" / "d1.png").write_bytes(b"png")
        (extracted_dir / "dogs" / "notes.txt").write_text("ignored")

        app.dependency_overrides[get_db] = lambda: db_session
        try:
            with patch("app.services.image_process.get_settings") as mock_settings:
                mock_settings.return_value.upload_folder = str(tmp_path)

                get_resp = client.get(f"/api/v1/data/process/image/{file_id}")
                assert get_resp.status_code == 200
                data = get_resp.json()["data"]
                assert data["image_size"] == 224
                assert data["batch_size"] == 32
                assert data["color_mode"] == "rgb"
                assert data["label_mode"] == "int"
                assert data["channels"] == 3
                assert data["suggested_input_shape"] == "224, 224, 3"
                assert data["num_classes"] == 2
                assert data["total_images"] == 3
                assert data["classes"] == [{"name": "cats", "count": 2}, {"name": "dogs", "count": 1}]

                patch_resp = client.patch(
                    f"/api/v1/data/process/image/{file_id}",
                    json={
                        "image_size": 64,
                        "batch_size": 16,
                        "color_mode": "grayscale",
                        "label_mode": "categorical",
                    },
                )
                assert patch_resp.status_code == 200
                updated = patch_resp.json()["data"]
                assert updated["image_size"] == 64
                assert updated["batch_size"] == 16
                assert updated["color_mode"] == "grayscale"
                assert updated["label_mode"] == "categorical"
                assert updated["channels"] == 1
                assert updated["suggested_input_shape"] == "64, 64, 1"

                persisted = db_session.get(ImageProperties, file_id)
                assert persisted is not None
                assert persisted.image_size == 64
                assert persisted.color_mode == "grayscale"
                assert persisted.label_mode == "categorical"
        finally:
            app.dependency_overrides.pop(get_db, None)

    def test_patch_image_properties_rejects_invalid_values(self, db_session):
        file_id = uuid4()
        db_session.add(DataFile(id=file_id, file_name="imgs.zip", file_type="zip", disk_name="imgs.zip"))
        db_session.commit()

        app.dependency_overrides[get_db] = lambda: db_session
        try:
            bad_color = client.patch(f"/api/v1/data/process/image/{file_id}", json={"color_mode": "cmyk"})
            assert bad_color.status_code == 422

            bad_size = client.patch(f"/api/v1/data/process/image/{file_id}", json={"image_size": 0})
            assert bad_size.status_code == 422

            bad_label = client.patch(f"/api/v1/data/process/image/{file_id}", json={"label_mode": "one_hot"})
            assert bad_label.status_code == 422
        finally:
            app.dependency_overrides.pop(get_db, None)
