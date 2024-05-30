import { render, screen } from "@testing-library/react";
import App from "../src/App";

test("renders learn react link", () => {
  render(<App />);
  const linkElement = screen.getByText(
    /TensorMap is a web application that enables you to create deep learning models using a graphical interface without having to know how to code. It is an open-source application./i,
  );
  expect(linkElement).toBeInTheDocument();
});
