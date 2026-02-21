"""seed_sample_project_pipeline

Revision ID: f4a8c2e91b37
Revises: 2e66fba94864
Create Date: 2026-02-23 22:00:00.000000

"""

import json
import os
import shutil

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f4a8c2e91b37"
down_revision = "2e66fba94864"
branch_labels = None
depends_on = None

# Fixed UUIDs for idempotency
PROJECT_UUID = "00000000-0000-4000-a000-000000000001"
FILE_UUID = "00000000-0000-4000-a000-000000000002"

# File paths (relative to tensormap-backend/, where alembic runs)
DATA_DIR = "./data"
MODEL_DIR = "./templates/json-model"
SOURCE_CSV = os.path.join(DATA_DIR, "test_dataset.csv")
TARGET_CSV = os.path.join(DATA_DIR, "iris_sample.csv")
MODEL_JSON_PATH = os.path.join(MODEL_DIR, "iris-classifier.json")


def upgrade():
    bind = op.get_bind()

    # 1. Remove old bare "Sample Project" if it has no associated files or models
    bind.execute(
        sa.text("""
            DELETE FROM project
            WHERE name = 'Sample Project'
              AND id NOT IN (SELECT DISTINCT project_id FROM data_file WHERE project_id IS NOT NULL)
              AND id NOT IN (SELECT DISTINCT project_id FROM model_basic WHERE project_id IS NOT NULL)
        """)
    )

    # 2. Insert project with fixed UUID
    bind.execute(
        sa.text("""
            INSERT INTO project (id, name, description)
            VALUES (:id, :name, :desc)
            ON CONFLICT (id) DO NOTHING
        """),
        {"id": PROJECT_UUID, "name": "Iris Sample Project", "desc": "Pre-configured Iris classification pipeline"},
    )

    # 3. Copy CSV dataset
    if os.path.exists(SOURCE_CSV) and not os.path.exists(TARGET_CSV):
        shutil.copy2(SOURCE_CSV, TARGET_CSV)

    # 4. Insert data_file
    bind.execute(
        sa.text("""
            INSERT INTO data_file (id, file_name, file_type, project_id)
            VALUES (:id, :name, :type, :proj)
            ON CONFLICT (id) DO NOTHING
        """),
        {"id": FILE_UUID, "name": "iris_sample", "type": "csv", "proj": PROJECT_UUID},
    )

    # 5. Insert data_process (target field assignment)
    bind.execute(
        sa.text("""
            INSERT INTO data_process (target, file_id)
            SELECT :target, :fid
            WHERE NOT EXISTS (SELECT 1 FROM data_process WHERE file_id = :fid)
        """),
        {"target": "species", "fid": FILE_UUID},
    )

    # 6. Insert model_basic with full training config
    result = bind.execute(
        sa.text("""
            INSERT INTO model_basic (model_name, file_id, project_id, model_type, target_field,
                                     training_split, optimizer, metric, epochs, loss)
            VALUES (:name, :fid, :pid, :mtype, :target, :split, :opt, :met, :epochs, :loss)
            ON CONFLICT (model_name) DO NOTHING
            RETURNING id
        """),
        {
            "name": "iris-classifier",
            "fid": FILE_UUID,
            "pid": PROJECT_UUID,
            "mtype": 1,  # ProblemType.CLASSIFICATION
            "target": "species",
            "split": 80.0,
            "opt": "adam",
            "met": "accuracy",
            "epochs": 10,
            "loss": "sparse_categorical_crossentropy",
        },
    )

    row = result.fetchone()
    if row is None:
        return  # Model already exists — skip configs and JSON generation
    model_id = row[0]

    # 7. Insert model_configs (flattened ReactFlow graph)
    configs = _build_model_configs()
    for param, value in configs:
        bind.execute(
            sa.text("""
                INSERT INTO model_configs (parameter, value, model_id)
                VALUES (:param, :val, :mid)
            """),
            {"param": param, "val": str(value), "mid": model_id},
        )

    # 8. Generate Keras JSON model definition
    _generate_keras_json()


def downgrade():
    bind = op.get_bind()

    # Delete seed data in reverse dependency order
    bind.execute(
        sa.text("""
            DELETE FROM model_configs WHERE model_id IN (
                SELECT id FROM model_basic WHERE model_name = 'iris-classifier'
            )
        """)
    )
    bind.execute(sa.text("DELETE FROM model_basic WHERE model_name = 'iris-classifier'"))
    bind.execute(sa.text("DELETE FROM data_process WHERE file_id = :fid"), {"fid": FILE_UUID})
    bind.execute(sa.text("DELETE FROM data_file WHERE id = :id"), {"id": FILE_UUID})
    bind.execute(sa.text("DELETE FROM project WHERE id = :id"), {"id": PROJECT_UUID})

    # Remove generated files
    for path in [TARGET_CSV, MODEL_JSON_PATH]:
        if os.path.exists(path):
            os.remove(path)

    # Re-seed the original bare "Sample Project" (only if none exists)
    bind.execute(
        sa.text("""
            INSERT INTO project (id, name, description)
            SELECT gen_random_uuid(), 'Sample Project', 'A sample project to get started with TensorMap'
            WHERE NOT EXISTS (SELECT 1 FROM project WHERE name = 'Sample Project')
        """)
    )


def _build_model_configs():
    """Return flattened (parameter, value) pairs for the ReactFlow graph.

    Replicates the flatten logic from model_save_service: flatten the graph
    dict with dot separators to produce key-value config rows.
    """
    from flatten_json import flatten

    graph = {
        "nodes": [
            {
                "id": "n0",
                "type": "custominput",
                "data": {"params": {"dim-1": "4"}},
                "position": {"x": 100, "y": 200},
            },
            {
                "id": "n1",
                "type": "customdense",
                "data": {"params": {"units": "16", "activation": "relu"}},
                "position": {"x": 100, "y": 400},
            },
            {
                "id": "n2",
                "type": "customdense",
                "data": {"params": {"units": "3", "activation": "softmax"}},
                "position": {"x": 100, "y": 600},
            },
        ],
        "edges": [
            {"id": "e0-1", "source": "n0", "target": "n1"},
            {"id": "e1-2", "source": "n1", "target": "n2"},
        ],
    }

    params = flatten(graph, separator=".")
    return [(k, v) for k, v in sorted(params.items()) if v is not None]


def _generate_keras_json():
    """Build a Keras model programmatically and write its JSON definition.

    Using tf.keras at migration time ensures the JSON format matches the
    installed TensorFlow version (Keras JSON is version-dependent).
    """
    import tensorflow as tf

    os.makedirs(MODEL_DIR, exist_ok=True)

    inp = tf.keras.Input(shape=(4,), name="n0")
    x = tf.keras.layers.Dense(16, activation="relu", name="n1")(inp)
    out = tf.keras.layers.Dense(3, activation="softmax", name="n2")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    with open(MODEL_JSON_PATH, "w") as f:
        f.write(json.dumps(json.loads(model.to_json())) + "\n")
