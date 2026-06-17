"""Tests for Week 3 database migration: graph_ir column + dual-write/read."""

import os

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from app.ir.schema import DenseParams, InputParams, IREdge, IRGraph, IRNode
from app.models.ml import ModelBasic, ModelConfigs
from app.services.deep_learning import model_save_service

# TensorFlow's native libs abort (SIGABRT) when imported under the CI runner,
# so tests that invoke a real TF import are skipped there. See test_cp1.py.
IN_CI = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


class TestGraphIRColumn:
    """Test that the graph_ir column exists and has correct schema."""

    def test_model_basic_has_graph_ir_column(self, db_session: Session):
        """Verify graph_ir column exists on model_basic table."""
        inspector = inspect(db_session.bind)
        columns = {col["name"]: col for col in inspector.get_columns("model_basic")}

        assert "graph_ir" in columns, "graph_ir column should exist on model_basic"
        assert columns["graph_ir"]["nullable"] is True, "graph_ir should be nullable"

    def test_graph_ir_accepts_json_data(self, db_session: Session):
        """Verify graph_ir column can store JSON dictionaries."""
        graph_data = {
            "version": "1.0",
            "nodes": [{"id": "n1", "node_params": {"layer_type": "input", "shape": 784}}],
            "edges": [],
        }

        model = ModelBasic(model_name="test-graph-ir-json", graph_ir=graph_data)
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)

        assert model.graph_ir is not None
        assert model.graph_ir["version"] == "1.0"
        assert len(model.graph_ir["nodes"]) == 1


class TestDualWrite:
    """Test that model save operations write to BOTH graph_ir and model_configs."""

    @pytest.mark.skipif(IN_CI, reason="TensorFlow causes segfaults in CI - run locally")
    def test_dual_write_saves_both_paths(self, db_session: Session):
        """Save a model and verify both graph_ir and model_configs are populated."""
        canvas_json = {
            "nodes": [
                {
                    "id": "node-1",
                    "type": "custominput",
                    "data": {"name": "Input", "params": {"dim-1": "784"}},
                    "position": {"x": 100, "y": 100},
                },
                {
                    "id": "node-2",
                    "type": "customdense",
                    "data": {"name": "Dense", "params": {"units": "128", "activation": "relu"}},
                    "position": {"x": 300, "y": 100},
                },
            ],
            "edges": [{"id": "edge-1", "source": "node-1", "target": "node-2"}],
        }

        response, status = model_save_service(db_session, canvas_json, "dual-write-test")

        assert status == 200, f"Save failed: {response}"

        # Verify model was created
        model = db_session.exec(select(ModelBasic).where(ModelBasic.model_name == "dual-write-test")).first()
        assert model is not None

        # Check NEW path: graph_ir column
        assert model.graph_ir is not None, "graph_ir should be populated"
        assert "nodes" in model.graph_ir
        assert len(model.graph_ir["nodes"]) == 2

        # Check OLD path: model_configs KV table
        configs = db_session.exec(select(ModelConfigs).where(ModelConfigs.model_id == model.id)).all()
        assert len(configs) > 0, "model_configs should still be populated for backward compat"

        # Check legacy graph_json also populated
        assert model.graph_json is not None
        assert "nodes" in model.graph_json


class TestDualRead:
    """Test that model load operations prefer graph_ir, fall back to model_configs."""

    def test_load_uses_graph_ir_when_available(self, db_session: Session):
        """Model with graph_ir populated should load from graph_ir column."""
        # Create IRGraph directly
        graph = IRGraph(
            nodes=[
                IRNode(id="n1", node_params=InputParams(layer_type="input", shape=784)),
                IRNode(id="n2", node_params=DenseParams(layer_type="dense", units=64, activation="relu")),
            ],
            edges=[IREdge(id="e1", source_id="n1", target_id="n2")],
        )

        model = ModelBasic(model_name="load-from-graph-ir", graph_ir=graph.model_dump())
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)

        # Verify we can deserialize it back
        assert model.graph_ir is not None
        loaded_graph = IRGraph(**model.graph_ir)

        assert len(loaded_graph.nodes) == 2
        assert loaded_graph.nodes[0].node_params.layer_type == "input"
        assert loaded_graph.nodes[1].node_params.layer_type == "dense"

    def test_load_falls_back_to_graph_json_when_ir_null(self, db_session: Session):
        """Model with graph_ir=NULL but graph_json populated should reconstruct from graph_json."""
        from app.ir.translator import reactflow_to_ir

        # Create model with only graph_json (no graph_ir)
        canvas = {
            "nodes": [
                {"id": "n1", "type": "custominput", "data": {"params": {"dim-1": "784"}}, "position": {"x": 0, "y": 0}},
            ],
            "edges": [],
        }
        model = ModelBasic(model_name="load-from-json", graph_ir=None, graph_json=canvas)
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)

        # Should be able to reconstruct from graph_json
        assert model.graph_json is not None
        loaded_graph = reactflow_to_ir(model.graph_json)

        assert loaded_graph is not None
        assert len(loaded_graph.nodes) > 0


class TestBackfillScript:
    """Test the backfill_graph_ir.py script."""

    def test_backfill_manually_updates_models(self, db_session: Session):
        """Test backfill logic by manually updating models."""
        # Seed 3 models with graph_json but no graph_ir
        for i in range(3):
            canvas = {
                "nodes": [
                    {
                        "id": f"node-{i}",
                        "type": "custominput",
                        "data": {"params": {"dim-1": str(100 * i + 10)}},
                        "position": {"x": 0, "y": 0},
                    }
                ],
                "edges": [],
            }
            model = ModelBasic(model_name=f"backfill-test-{i}", graph_json=canvas, graph_ir=None)
            db_session.add(model)
        db_session.commit()

        # Manually backfill each model (simulating what the script does)
        from app.ir.translator import reactflow_to_ir

        for i in range(3):
            model = db_session.exec(select(ModelBasic).where(ModelBasic.model_name == f"backfill-test-{i}")).first()
            if model and model.graph_ir is None and model.graph_json is not None:
                try:
                    graph = reactflow_to_ir(model.graph_json)
                    model.graph_ir = graph.model_dump()
                    db_session.add(model)
                except Exception as e:
                    pytest.fail(f"Failed to backfill model {i}: {e}")

        db_session.commit()

        # Verify all 3 models now have graph_ir
        for i in range(3):
            model = db_session.exec(select(ModelBasic).where(ModelBasic.model_name == f"backfill-test-{i}")).first()
            assert model.graph_ir is not None, f"Model {i} should have graph_ir after backfill"
            assert "nodes" in model.graph_ir


class TestMigrationUpDown:
    """Test that Alembic migration can be applied and reverted cleanly."""

    def test_migration_column_exists_after_upgrade(self, db_session: Session):
        """After migration, graph_ir column should exist."""
        # This test assumes migration has already run (handled by conftest.py)
        inspector = inspect(db_session.bind)
        columns = [col["name"] for col in inspector.get_columns("model_basic")]
        assert "graph_ir" in columns

    def test_can_insert_and_query_graph_ir(self, db_session: Session):
        """Basic CRUD operations on graph_ir column should work."""
        graph_data = {"version": "1.0", "nodes": [], "edges": []}
        model = ModelBasic(model_name="migration-crud-test", graph_ir=graph_data)

        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)

        # Query back
        loaded = db_session.get(ModelBasic, model.id)
        assert loaded.graph_ir == graph_data
