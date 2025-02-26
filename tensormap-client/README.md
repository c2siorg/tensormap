```markdown
# TensorMap Frontend

This repository contains the frontend code for TensorMap, a web application that allows users to create machine learning algorithms visually.

## Overview

The frontend of TensorMap is built using ReactJS. It provides a user-friendly interface for:

* **Data Upload:** Adding datasets to TensorMap.
* **Data Preprocessing:** Cleaning, transforming, and preparing data for model training.
* **Model Design:** Visually designing machine learning models using a drag-and-drop interface.
* **Model Execution:** Training and evaluating models with real-time feedback.
* **Code Generation:** Generating production-ready code in Python or JavaScript.

## Installation Instructions

To run the TensorMap Frontend App, you need to have Node.js and NPM (Node Package Manager) installed on your system.

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/c2siorg/tensormap.git](https://github.com/c2siorg/tensormap.git)
   cd tensormap/tensormap-client 
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**

   ```bash
   npm start
   ```

4. **Access the app:** Open your web browser and go to `http://localhost:3000`.

## Usage

1. **Open the TensorMap Frontend App** in your web browser.
2. **Add a new dataset** to TensorMap using the Data Upload interface.
3. **Preprocess the data** in the Data Processing tab.
4. **Drag and drop components** from the sidebar onto the canvas to create your model.
5. **Edit the properties** of the selected component on the canvas to customize your algorithm.
6. **Preview your algorithm** in the preview window to see how it will perform.

## Testing

The frontend uses `Jest` for testing. All configurations and sample tests are located in the `tests` directory.

To run the tests:

```bash
npm run test
```

## Technology Stack

* **ReactJS:** A JavaScript library for building user interfaces.
* **Semantic UI React:** A UI component library for styling and layout.
* **React Router:** A library for handling navigation between different views.
* **Axios:** A library for making HTTP requests to the backend.
* **Recoil:** A state management library for React.

## Component Hierarchy

The frontend components are organized in a hierarchical structure:

* **App:** The root component that manages routing and overall layout.
* **Containers:** Top-level components for different views (Home, Data Upload, Data Preprocessing, Model Design, Model Execution, Code Generation).
* **Components:** Reusable UI elements used within containers (buttons, forms, charts, graphs).

## State Management

Recoil is used for managing application state. It provides a centralized store for data and allows components to subscribe to and update state changes.

## Contributing

Please read the 'Note to Contributors' in the project wiki for more details and the 'Contributing.md' file.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
```