"""uuid_primary_keys

Revision ID: a1b2c3d4e5f6
Revises: b8d925d66e78
Create Date: 2026-02-22 15:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "b8d925d66e78"
branch_labels = None
depends_on = None


def upgrade():
    # Enable uuid-ossp extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # Truncate all tables (dev environment, data loss acceptable)
    op.execute(
        "TRUNCATE model_results, model_configs, model_basic, image_properties, data_process, data_file, project CASCADE"
    )

    # Drop FK constraints first
    op.drop_constraint("model_basic_project_id_fkey", "model_basic", type_="foreignkey")
    op.drop_constraint("model_basic_file_id_fkey", "model_basic", type_="foreignkey")
    op.drop_constraint("data_file_project_id_fkey", "data_file", type_="foreignkey")
    op.drop_constraint("data_process_file_id_fkey", "data_process", type_="foreignkey")
    op.drop_constraint("image_properties_id_fkey", "image_properties", type_="foreignkey")

    # Alter project.id from INTEGER to UUID
    op.execute("ALTER TABLE project ALTER COLUMN id DROP DEFAULT")
    op.execute("ALTER TABLE project ALTER COLUMN id SET DATA TYPE UUID USING uuid_generate_v4()")
    op.execute("ALTER TABLE project ALTER COLUMN id SET DEFAULT uuid_generate_v4()")

    # Alter data_file.id from INTEGER to UUID
    op.execute("ALTER TABLE data_file ALTER COLUMN id DROP DEFAULT")
    op.execute("ALTER TABLE data_file ALTER COLUMN id SET DATA TYPE UUID USING uuid_generate_v4()")
    op.execute("ALTER TABLE data_file ALTER COLUMN id SET DEFAULT uuid_generate_v4()")

    # Alter data_file.project_id from INTEGER to UUID
    op.execute("ALTER TABLE data_file ALTER COLUMN project_id SET DATA TYPE UUID USING NULL::UUID")

    # Alter data_process.file_id from INTEGER to UUID
    op.execute("ALTER TABLE data_process ALTER COLUMN file_id SET DATA TYPE UUID USING uuid_generate_v4()")

    # Alter image_properties.id from INTEGER to UUID
    op.execute("ALTER TABLE image_properties ALTER COLUMN id SET DATA TYPE UUID USING uuid_generate_v4()")

    # Alter model_basic.file_id from INTEGER to UUID
    op.execute("ALTER TABLE model_basic ALTER COLUMN file_id SET DATA TYPE UUID USING uuid_generate_v4()")

    # Alter model_basic.project_id from INTEGER to UUID
    op.execute("ALTER TABLE model_basic ALTER COLUMN project_id SET DATA TYPE UUID USING NULL::UUID")

    # Re-create FK constraints
    op.create_foreign_key("data_file_project_id_fkey", "data_file", "project", ["project_id"], ["id"])
    op.create_foreign_key("data_process_file_id_fkey", "data_process", "data_file", ["file_id"], ["id"])
    op.create_foreign_key("image_properties_id_fkey", "image_properties", "data_file", ["id"], ["id"])
    op.create_foreign_key("model_basic_file_id_fkey", "model_basic", "data_file", ["file_id"], ["id"])
    op.create_foreign_key("model_basic_project_id_fkey", "model_basic", "project", ["project_id"], ["id"])


def downgrade():
    # Drop FK constraints
    op.drop_constraint("model_basic_project_id_fkey", "model_basic", type_="foreignkey")
    op.drop_constraint("model_basic_file_id_fkey", "model_basic", type_="foreignkey")
    op.drop_constraint("image_properties_id_fkey", "image_properties", type_="foreignkey")
    op.drop_constraint("data_process_file_id_fkey", "data_process", type_="foreignkey")
    op.drop_constraint("data_file_project_id_fkey", "data_file", type_="foreignkey")

    # Truncate all tables
    op.execute(
        "TRUNCATE model_results, model_configs, model_basic, image_properties, data_process, data_file, project CASCADE"
    )

    # Revert columns back to INTEGER
    op.execute("ALTER TABLE project ALTER COLUMN id SET DATA TYPE INTEGER USING 0")
    op.execute("ALTER TABLE data_file ALTER COLUMN id SET DATA TYPE INTEGER USING 0")
    op.execute("ALTER TABLE data_file ALTER COLUMN project_id SET DATA TYPE INTEGER USING NULL::INTEGER")
    op.execute("ALTER TABLE data_process ALTER COLUMN file_id SET DATA TYPE INTEGER USING 0")
    op.execute("ALTER TABLE image_properties ALTER COLUMN id SET DATA TYPE INTEGER USING 0")
    op.execute("ALTER TABLE model_basic ALTER COLUMN file_id SET DATA TYPE INTEGER USING 0")
    op.execute("ALTER TABLE model_basic ALTER COLUMN project_id SET DATA TYPE INTEGER USING NULL::INTEGER")

    # Re-create FK constraints
    op.create_foreign_key("data_file_project_id_fkey", "data_file", "project", ["project_id"], ["id"])
    op.create_foreign_key("data_process_file_id_fkey", "data_process", "data_file", ["file_id"], ["id"])
    op.create_foreign_key("image_properties_id_fkey", "image_properties", "data_file", ["id"], ["id"])
    op.create_foreign_key("model_basic_file_id_fkey", "model_basic", "data_file", ["file_id"], ["id"])
    op.create_foreign_key("model_basic_project_id_fkey", "model_basic", "project", ["project_id"], ["id"])
