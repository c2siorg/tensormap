[![Build Status](https://travis-ci.com/scorelab/TensorMap.svg?branch=master)](https://travis-ci.com/scorelab/TensorMap)  
[![Join the chat at https://gitter.im/scorelab/TensorMap](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/scorelab/TensorMap)  

# TensorMap

## Table of Contents

1.  [**Introduction**](#1-introduction)
    * [Overview of TensorMap](#11-overview-of-tensormap)
    * [Key Features and Benefits](#12-key-features-and-benefits)
    * [Target Audience](#13-target-audience)
2.  [**Installation and Setup**](#2-installation-and-setup)
    * [Prerequisites](#21-prerequisites)
    * [Docker-based Installation](#22-docker-based-installation)
    * [Manual Installation (Advanced)](#23-manual-installation-advanced)
    * [Troubleshooting](#24-troubleshooting)
3.  [**User Interface**](#3-user-interface)
    * [Home Screen](#31-home-screen)
    * [Data Upload](#32-data-upload)
    * [Data Preprocessing](#33-data-preprocessing)
    * [Model Design](#34-model-design)
    * [Model Execution](#35-model-execution)
    * [Code Generation](#36-code-generation)
4.  [**Backend Architecture**](#4-backend-architecture)
    * [Technology Stack](#41-technology-stack)
    * [API Endpoints](#42-api-endpoints)
    * [Database Schema](#43-database-schema)
5.  [**Frontend Architecture**](#5-frontend-architecture)
    * [Technology Stack](#51-technology-stack)
    * [Component Hierarchy](#52-component-hierarchy)
    * [State Management](#53-state-management)
6.  [**Advanced Usage**](#6-advanced-usage)
    * [Custom Model Architectures](#61-custom-model-architectures)
    * [Hyperparameter Tuning](#62-hyperparameter-tuning)
    * [Integration with External Tools](#63-integration-with-external-tools)
7.  [**Contributing**](#7-contributing)
    * [Bug Reports and Feature Requests](#71-bug-reports-and-feature-requests)
    * [Code Contributions](#72-code-contributions)
    * [Community Guidelines](#73-community-guidelines)
8. [**API Documentation**](#8-api-documentation)


## 1. Introduction

### 1.1 Overview of TensorMap

TensorMap is a web-based platform designed to democratize machine learning by providing a visual and intuitive environment for building, experimenting with, and deploying machine learning models. It eliminates the need for extensive coding knowledge, making it accessible to beginners and domain experts alike.

TensorMap leverages the power of TensorFlow, a leading open-source machine learning framework, while abstracting away its complexities through a user-friendly graphical interface. Users can drag and drop components, connect them to form model architectures, and configure parameters without writing a single line of code.

### 1.2 Key Features and Benefits

* **Visual Model Design:** Create complex machine learning models with ease using a drag-and-drop interface.
* **No Coding Required:** Build and experiment with models without any prior programming experience.
* **Data Preprocessing:** Clean, transform, and prepare data for model training using built-in tools.
* **Model Execution:** Train and evaluate models with real-time feedback and visualizations.
* **Code Generation:** Generate production-ready code in popular languages (Python, JavaScript) for seamless deployment.
* **Extensible Architecture:** Supports custom model components and integrations with external tools.

### 1.3 Target Audience

TensorMap caters to a wide range of users, including:

* **Students and Educators:** Learn and teach machine learning concepts in an engaging and interactive way.
* **Data Scientists and Analysts:** Prototype and experiment with models quickly without getting bogged down in code.
* **Domain Experts:** Apply machine learning to their respective fields without needing deep technical expertise.
* **Machine Learning Enthusiasts:** Explore and experiment with different model architectures and algorithms.

## 2. Installation and Setup

### 2.1 Prerequisites

Before installing TensorMap, ensure you have the following prerequisites installed on your system:

* **Docker:** A containerization platform for packaging and running applications.
* **Docker Compose:** A tool for defining and managing multi-container Docker applications.

### 2.2 Docker-based Installation

The recommended way to install TensorMap is using Docker. Follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/c2siorg/tensormap.git](https://github.com/c2siorg/tensormap.git)
    cd tensormap
    ```

2.  **Build and run the application:**

    ```bash
    docker-compose up --build
    ```

    This command will:

    * Start a PostgreSQL database container.
    * Build and run the TensorMap server (Flask backend) container.
    * Build and run the TensorMap client (React frontend) container.

3.  **Access the application:**

    * **Frontend:** Open your web browser and go to `http://localhost:5173`.
    * **Backend:** The Flask API will be available at `http://localhost:5000`.

### 2.3 Manual Installation (Advanced)

For advanced users who prefer manual installation, follow these steps:

1.  **Install PostgreSQL:** Download and install PostgreSQL on your system.
2.  **Create a database:** Create a new database in PostgreSQL and note down the database name, username, and password.
3.  **Install Python dependencies:** Create a virtual environment and install the required Python packages listed in `requirements.txt`.
4.  **Configure environment variables:** Set the following environment variables:
    * `DATABASE_URL`: Connection string for the PostgreSQL database.
    * `FLASK_APP`: Path to the Flask application file (`app.py`).
    * `FLASK_ENV`: Set to `development` for development mode.
5.  **Run the Flask backend:** Execute `flask run` to start the backend server.
6.  **Install Node.js dependencies:** Navigate to the `tensormap-client` directory and run `npm install` to install the frontend dependencies.
7.  **Run the React frontend:** Execute `npm start` to start the frontend development server.

### 2.4 Troubleshooting

If you encounter any issues during installation, refer to the troubleshooting section in the documentation or seek assistance from the community forum.

## 3. User Interface

### 3.1 Home Screen

The home screen provides an overview of TensorMap's features and capabilities. It also offers quick links to key functionalities:

* **Upload Data:** Navigate to the data upload section.
* **Create New Project:** Start a new machine learning project.
* **Contribute:** Contribute to the TensorMap project on GitHub.

### 3.2 Data Upload

The data upload section allows users to upload datasets in various formats, including CSV, JSON, and image files. Users can also preview the uploaded data and select relevant features for model training.

### 3.3 Data Preprocessing

The data preprocessing section provides tools for cleaning, transforming, and preparing data for model training. Users can perform operations such as:

* **Data Cleaning:** Handle missing values, outliers, and inconsistencies.
* **Feature Scaling:** Normalize or standardize features to improve model performance.
* **Feature Engineering:** Create new features from existing ones.
* **Data Visualization:** Visualize data distributions and relationships using charts and graphs.

### 3.4 Model Design

The model design section is the heart of TensorMap. It features a drag-and-drop canvas where users can visually construct model architectures. Users can:

* **Drag and Drop Components:** Add various layers, activation functions, and other components to the canvas.
* **Connect Components:** Connect components to define the flow of data through the model.
* **Configure Parameters:** Customize the properties of each component, such as the number of neurons in a layer or the learning rate of an optimizer.
* **Preview Model:** Visualize the model architecture and its connections.

### 3.5 Model Execution

Once the model is designed, users can train and evaluate it using the model execution section. This section provides:

* **Training Controls:** Start, stop, and monitor the training process.
* **Performance Metrics:** Track metrics such as accuracy, loss, and precision in real-time.
* **Visualizations:** Visualize training progress and model performance using charts and graphs.

### 3.6 Code Generation

After training and evaluating the model, users can generate production-ready code in their preferred language (Python, JavaScript). The generated code includes:

* **Model Architecture:** Code to define the model structure and its components.
* **Data Preprocessing:** Code to preprocess data before feeding it to the model.
* **Model Training:** Code to train the model using the specified parameters.
* **Model Evaluation:** Code to evaluate the model's performance on test data.

## 4. Backend Architecture

### 4.1 Technology Stack

The TensorMap backend is built using the following technologies:

* **Flask:** A lightweight Python web framework for building APIs.
* **PostgreSQL:** A relational database for storing data and model configurations.
* **TensorFlow:** An open-source machine learning framework for model execution.

### 4.2 API Endpoints

The backend exposes a RESTful API with the following endpoints:

* **Data Upload:**
    * `/upload`: Upload data files.
    * `/datasets`: Get a list of uploaded datasets.
* **Data Preprocessing:**
    * `/preprocess`: Apply preprocessing operations to data.
    * `/visualize`: Generate visualizations from data.
* **Model Design:**
    * `/models`: Create, retrieve, update, and delete models.
    * `/components`: Get a list of available model components.
* **Model Execution:**
    * `/train`: Train a model.
    * `/evaluate`: Evaluate a model.
* **Code Generation:**
    * `/generate`: Generate code for a model.

### 4.3 Database Schema

The database schema consists of the following tables:

* **Datasets:** Stores information about uploaded datasets.
* **Models:** Stores model configurations and architectures.
* **Components:** Stores information about available model components.
* **Training Logs:** Stores logs and metrics from model training runs.

## 5. Frontend Architecture

### 5.1 Technology Stack

The TensorMap frontend is built using the following technologies:

* **ReactJS:** A JavaScript library for building user interfaces.
* **Semantic UI React:** A UI component library for styling and layout.
* **React Router:** A library for handling navigation between different views.
* **Axios:** A library for making HTTP requests to the backend.
* **Recoil:** A state management library for React.

### 5.2 Component Hierarchy

The frontend components are organized in a hierarchical structure:

* **App:** The root component that manages routing and overall layout.
* **Containers:** Top-level components for different views (Home, Data Upload, Data Preprocessing, Model Design, Model Execution, Code Generation).
* **Components:** Reusable UI elements used within containers (buttons, forms, charts, graphs).

### 5.3 State Management

Recoil is used for managing application state. It provides a centralized store for data and allows components to subscribe to and update state changes.

## 6. Advanced Usage

### 6.1 Custom Model Architectures

TensorMap allows users to define custom model architectures by combining existing components or creating new ones. This enables users to experiment with novel model designs and explore cutting-edge research.

### 6.2 Hyperparameter Tuning

Users can fine-tune model performance by adjusting hyperparameters such as learning rate, batch size, and number of epochs. TensorMap provides tools for visualizing the impact of hyperparameter changes on model performance.

### 6.3 Integration with External Tools

TensorMap can be integrated with external tools and services to enhance its functionality. For example, users can connect to cloud storage providers to access larger datasets or integrate with model deployment platforms for seamless productionization.

## 7. Contributing

### 7.1 Bug Reports and Feature Requests

If you encounter any bugs or have suggestions for new features, please submit an issue on the GitHub repository. Provide detailed information about the issue or suggestion, including steps to reproduce the bug or a clear description of the desired feature.

### 7.2 Code Contributions

We welcome code contributions from the community. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Ensure your code adheres to the project's coding style and conventions.

### 7.3 Community Guidelines

We encourage a welcoming and inclusive community. Please be respectful of others and adhere to the project's code of conduct.

## 8. API Documentation

**Base URL:** http://localhost:5000/api/v1 (This can be configured in `config.yaml`)

**Authentication:** (If applicable, describe the authentication method used)

**Error Handling:**

* The API uses standard HTTP status codes to indicate success or failure of requests.
* Error responses will include a JSON object with a `message` field describing the error.

**Endpoints**

**1. Data Upload** (`tensormap-server/endpoints/DataUpload/urls.py`)

* **POST /upload/file**
    * Uploads a data file to the server.
    * Request Body:
        * `file`: The data file (multipart/form-data).
    * Response:
        * 201 Created: File uploaded successfully.
        * 400 Bad Request: Invalid file format or other error.
* **GET /upload/file/<int:file\_id>**
    * Retrieves information about an uploaded file.
    * Response:
        * 200 OK: JSON object with file details (name, type, size, etc.).
        * 404 Not Found: File not found.
* **DELETE /upload/file/<int:file\_id>**
    * Deletes an uploaded file.
    * Response:
        * 204 No Content: File deleted successfully.
        * 404 Not Found: File not found.

**2. Data Preprocessing** (`tensormap-server/endpoints/DataProcess/urls.py`)

* **POST /process/target**
    * (Purpose unclear from the code. Likely sets or updates a target variable for a dataset.)
    * Request Body: (Details needed based on the actual implementation)
    * Response: (Details needed based on the actual implementation)
* **GET /process/target/<int:file\_id>**
    * (Purpose unclear from the code. Likely retrieves the target variable for a dataset.)
    * Response: (Details needed based on the actual implementation)
* **PUT /process/target/<int:file\_id>**
    * (Purpose unclear from the code. Likely updates the target variable for a dataset.)
    * Request Body: (Details needed based on the actual implementation)
    * Response: (Details needed based on the actual implementation)
* **DELETE /process/target/<int:file\_id>**
    * (Purpose unclear from the code. Likely removes the target variable for a dataset.)
    * Response: (Details needed based on the actual implementation)
* **GET /process/data\_metrics/<int:file\_id>**
    * Retrieves data metrics for a dataset.
    * Response:
        * 200 OK: JSON object with data metrics (e.g., correlation matrix, data types).
        * 404 Not Found: Dataset not found.
* **GET /process/file/<int:file\_id>**
    * Retrieves file data for a dataset.
    * Response:
        * 200 OK: JSON object with file data.
        * 404 Not Found: Dataset not found.
* **POST /process/preprocess/<int:file\_id>**
    * Applies preprocessing operations to a dataset.
    * Request Body: (Details needed based on the actual implementation)
    * Response: (Details needed based on the actual implementation)
* **GET /process/image\_preprocess/<int:file\_id>**
    * (Purpose unclear from the code. Likely retrieves image preprocessing properties for a dataset.)
    * Response: (Details needed based on the actual implementation)

**3. Deep Learning** (`tensormap-server/endpoints/DeepLearning/urls.py`)

* **POST /model/validate**
    * Validates a model design.
    * Request Body: (Details needed based on the actual implementation)
    * Response:
        * 200 OK: Model is valid.
        * 400 Bad Request: Model is invalid, with details about the errors.
* **POST /model/code**
    * Generates code for a model.
    * Request Body: (Details needed based on the actual implementation)
    * Response:
        * 200 OK: JSON object with generated code.
        * 400 Bad Request: Invalid model design or other error.
* **GET /model/get\_model\_list**
    * Retrieves a list of available models.
    * Response:
        * 200 OK: JSON array of model names or IDs.
* **POST /model/run**
    * Runs a model.
    * Request Body: (Details needed based on the actual implementation)
    * Response:
        * 200 OK: JSON object with model execution results.
        * 400 Bad Request: Invalid model or input data.
