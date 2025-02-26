```markdown
# TensorMap Backend

This document provides information on setting up and running the backend for TensorMap, a web application that allows users to visually create machine learning algorithms.

## Prerequisites

* **PostgreSQL server:** Install a PostgreSQL server. You can download and install it from the official website.
* **Python 3.x:** Ensure you have Python 3.x installed (Python 3.9 is recommended).

## Starting the Server and Database using Docker

The easiest way to get started is using Docker. 

**Requirements:**

* **Docker Compose:** Install Docker Compose.
* **Docker:** Install Docker.

**Run with Docker Compose:**

```bash
docker compose up
```

This will start both the PostgreSQL database and the Flask backend server.

## Starting PostgreSQL using Docker (Alternative)

If you prefer to run only the PostgreSQL database in Docker:

```bash
docker run -d \
  --name database \
  -e POSTGRES_DB='your_database_name' \
  -e POSTGRES_USER='your_database_user' \
  -e POSTGRES_PASSWORD='your_database_password' \
  -p 5432:5432 \
  postgres
```

Replace `your_database_name`, `your_database_user`, and `your_database_password` with your desired credentials.

## Manual Installation Instructions

1. **Install Python dependencies:**
   * Create a virtual environment: Follow this guide to set up a virtual environment.
   * Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

2. **Set up the database:**
   * Create a PostgreSQL database and note the database name, username, and password.
   * Create a `.env` file in the `tensormap-server` directory and add the following:

     ```
     secret_key = 'your_secret_key' 
     db_name = 'your_database_name'
     db_host = 'your_database_host'
     db_password = 'your_database_password'
     db_user = 'your_database_user'
     ```

     Replace the placeholders with your actual database details.

3. **Set up the Flask app:**

   ```bash
   export FLASK_APP=app.py
   ```

4. **Run the backend:**

   ```bash
   flask run
   ```

## Application Architecture

The backend application has the following structure:

```
tensormap-server/
├── README.md
├── app.py                      # Main application file
├── .env                        # Environment variables
├── .gitignore
├── config.yaml                 # Configuration file
├── endpoints/                  # API endpoints
│   ├── DataProcess/
│   ├── DataUpload/
│   └── DeepLearning/
├── requirements.txt            # Python dependencies
├── migrations/                 # Database migrations
│   ├── README
│   ├── __pycache__
│   │   └── env.cpython-39.pyc
│   ├── alembic.ini
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── db_migration_table_v1.py
├── setup/                      # Setup and utilities
│   ├── settings.py
│   └── urls.py
├── shared/                     # Shared modules
│   ├── request/
│   │   └── response.py
│   ├── services/
│   │   └── config.py
│   └── utils.py
├── static/
├── templates/                  # HTML templates (if any)
└── tests/                      # Unit tests
    └── unit/
        └── test_DataUpload.py 
```


## Setting up the Database Locally

1. **Initialize the database:**

   ```bash
   flask db init
   ```

2. **Migrate the database:**

   ```bash
   flask db migrate
   ```

3. **Apply changes to the database:**

   ```bash
   flask db upgrade
   ```

## Database Modifications

If you modify database models, you need to generate and apply migrations:

1. **Generate migration scripts:**

   ```bash
   flask db migrate
   ```

2. **Apply migrations to the database:**

   ```bash
   flask db upgrade
   ```

3. **Commit the generated migration scripts** to version control.

## Testing

The backend uses `PyTest` for testing. Sample unit tests and configurations are in the `tests` directory.

To run backend tests:

```bash
pytest .
```

For database tests, create a mock database and add a sample data file using the `db_session` and `add_sample_file` fixtures in `conftest.py`.

## Further Documentation

* **API Endpoints:** See the API documentation for details on the available endpoints and their usage.
* **Contributing:** Refer to the contributing guidelines for information on how to contribute to the project.
```