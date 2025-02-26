---

[![Build Status](https://travis-ci.com/scorelab/TensorMap.svg?branch=master)](https://travis-ci.com/scorelab/TensorMap)  
[![Join the chat at https://gitter.im/scorelab/TensorMap](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/scorelab/TensorMap)  
[![HitCount](http://hits.dwyl.com/scorelab/TensorMap.svg)](http://hits.dwyl.com/scorelab/TensorMap)

# TensorMap
=======

## 🌟 Overview

TensorMap is a web application that allows users to create machine learning algorithms visually. TensorMap supports reverse engineering of the visual layout to a TensorFlow implementation in preferred languages. The goal of the project is to let beginners play with machine learning algorithms in TensorFlow without requiring extensive background knowledge about the library. For more details about the project, read our [project wiki](https://github.com/scorelab/TensorMap/wiki).

---

## 🚀 Key Features

- Drag-and-drop interface for neural network design

- Auto-generation of TensorFlow code (Python/JavaScript)

- Model visualization and version control

- Export capabilities for trained models

- Collaborative workspace support

---

## Getting Started

Follow these steps to set up and run TensorMap using Docker.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

- [Docker Compose](https://docs.docker.com/compose/install/)

---

### System Architecture

This repository has the following structure:

```
TensorMap/
├── tensormap-server/  # Backend services
├── tensormap-client/  # Frontend interface
├── docs/              # Documentation
└── scripts/           # Deployment 
```
---

### Running TensorMap with Docker

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/c2siorg/tensormap.git
   cd TensorMap
   ```

2. **Set Up Environment**:

   - Ensure Docker and Docker Compose are installed and running on your machine.

3. **Build and Run the Application**:

   Use Docker Compose to build and start the TensorMap services (database, server, and client):

   ```bash
   docker-compose up --build
   ```
   This will:
   - Start a PostgreSQL database.

   - Build and run the TensorMap server (Flask backend).

   - Build and run the TensorMap client (React frontend).

4. **Access the Application**:

   - **Frontend (Client)**: Open your browser and go to `http://localhost:5173`.

   - **Backend (Server)**: The Flask API will be available at `http://localhost:5000`.

5. **Stop the Application**:

   To stop the running services, press `Ctrl+C` in the terminal or run:

   ```bash
   docker-compose down
   ```

---

### Docker Compose Configuration

The `docker-compose.yml` file defines the following services:

- **Database**: PostgreSQL database for storing application data.

- **Server**: Flask backend for TensorMap.

- **Client**: React frontend for TensorMap.

You can modify the `docker-compose.yml` file to customize the setup (e.g., change ports or environment variables).

---

### Development with Docker

If you're developing TensorMap, you can use Docker to streamline your workflow:

- **Rebuild and Restart the Client**:

  ```bash
  docker-compose up --build client
  ```

- **View Logs**:

  ```bash
  docker-compose logs client
  ```

- **Access the Container Shell**:
  ```bash
  docker exec -it <client-container-id> /bin/sh
  ```
---

## Development Workflow

### Branching Strategy

```
git checkout -b feat/new-layer-type   # Feature development
git checkout -b fix/issue-123         # Bug fixes
git checkout -b docs/readme-update    # Documentation i
```

### Testing 

```
# Run backend tests
cd tensormap-server
pytest

# Run frontend tests
cd tensormap-client
npm test
```



### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

---
### Running TensorMap with Docker
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/c2siorg/tensormap.git
   cd TensorMap
   ```
2. **Set Up Environment**:
   - Ensure Docker and Docker Compose are installed and running on your machine.
3. **Build and Run the Application**:
   Use Docker Compose to build and start the TensorMap services (database, server, and client):
   ```bash
   docker-compose up --build
   ```
   This will:
   - Start a PostgreSQL database.
   - Build and run the TensorMap server (Flask backend).
   - Build and run the TensorMap client (React frontend).
4. **Access the Application**:
   - **Frontend (Client)**: Open your browser and go to `http://localhost:5173`.
   - **Backend (Server)**: The Flask API will be available at `http://localhost:5000`.
5. **Stop the Application**:
   To stop the running services, press `Ctrl+C` in the terminal or run:
   ```bash
   docker-compose down
   ```

---
### Docker Compose Configuration
The `docker-compose.yml` file defines the following services:
- **Database**: PostgreSQL database for storing application data.
- **Server**: Flask backend for TensorMap.
- **Client**: React frontend for TensorMap.
You can modify the `docker-compose.yml` file to customize the setup (e.g., change ports or environment variables).


---
### Development with Docker
If you're developing TensorMap, you can use Docker to streamline your workflow:
- **Rebuild and Restart the Client**:
  ```bash
  docker-compose up --build client
  ```
- **View Logs**:
  ```bash
  docker-compose logs client
  ```
- **Access the Container Shell**:
  ```bash
  docker exec -it <client-container-id> /bin/sh
  ```

---
## Running Test Cases
To ensure the quality and correctness of the codebase, TensorMap includes unit tests for both the backend (Flask server) and frontend (React client). Follow these steps to run the test cases locally:

### Backend Tests (Flask Server)
1. **Install Dependencies**:
   If you’re not using Docker, install the required dependencies using Poetry:
   ```bash
   poetry install
   ```
2. **Run Tests**:
   Use `pytest` to execute the backend test suite:
   ```bash
   poetry run pytest tests/
   ```
   Replace `tests/` with the path to your test directory if it differs.

3. **Test Coverage**:
   To measure test coverage, use the `pytest-cov` plugin:
   ```bash
   poetry run pytest --cov=app --cov-report=term-missing --cov-report=html tests/
   ```
   - This command generates a coverage report in the terminal and an interactive HTML report in the `htmlcov/` directory.
   - Open the HTML report:
     ```bash
     open htmlcov/index.html  # macOS
     xdg-open htmlcov/index.html  # Linux
     ```

### Frontend Tests (React Client)
1. **Navigate to the Client Directory**:
   ```bash
   cd client
   ```
2. **Install Dependencies**:
   Install the required dependencies using `npm` or `yarn`:
   ```bash
   npm install
   ```
3. **Run Tests**:
   Use `npm test` or `yarn test` to execute the frontend test suite:
   ```bash
   npm test
   ```
   This will run all Jest-based tests and display the results in the terminal.

4. **Coverage Report**:
   Generate a test coverage report for the frontend:
   ```bash
   npm test -- --coverage
   ```
   - A coverage report will be generated in the `coverage/` directory.
   - Open the HTML report:
     ```bash
     open coverage/lcov-report/index.html  # macOS
     xdg-open coverage/lcov-report/index.html  # Linux
     ```

---
## Continuous Integration (CI)
TensorMap uses GitHub Actions for continuous integration. The CI pipeline automatically runs the following tasks on every push or pull request:
- Runs backend and frontend tests.
- Generates test coverage reports.
- Checks code quality using linters like `ruff`.

To view the CI status, check the [Actions tab](https://github.com/scorelab/TensorMap/actions) in the repository.

---

---
=======
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/scorelab/TensorMap/blob/master/LICENSE) file for details.

---

This updated `README.md` includes clear instructions for running TensorMap using Docker, making it easier for users to get started. Let me know if you need further adjustments!

