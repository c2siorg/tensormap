## Frontend setup and architecture

========

TensorMap is a web application that allows users to create machine learning algorithms visually. The front-end of TensorMap is built using ReactJS. This documentation provides an overview of the different components of the Tensor Map front-end app and how they work.

### Installation instructions

To use the TensorMap Frontend App, you need to have Node.js (>= 22.12.0) and NPM installed on your system. Follow the steps below to install and run the TensorMap Frontend App:

1.  Install the dependencies by running `npm install` in the project directory.
2.  Start the development server by running `npm start` in the project directory.
3.  Open [http://localhost:3300](http://localhost:3300/) in your web browser to access the TensorMap app.

### Usage

Follow the steps below to create your machine learning algorithm:

1.  Open the TensorMap Frontend App in your web browser.
2.  Add New Dataset to the TensorMap by using Data Upload Interface.
3.  Preprocess the data in the Data Processing Tab.
4.  Drag and drop components from the sidebar onto the canvas to create your model.
5.  Edit the properties of the selected component on the canvas to customize your algorithm.
6.  Preview your algorithm in the preview window to see how it will perform.

## Testing

For the front end, `Vitest` is used for testing. Test suites are organized beside their components under `tests/`, `__tests__/`, or inside `src/test/`. To run the tests, run `npm test` (or `npm run test:watch` for watch mode) in the terminal.
