import { render, screen } from "@testing-library/react";
import DropoutNode from "./DropoutNode";
import { ReactFlowProvider } from "reactflow";

const wrapper = ({ children }) => (
  <ReactFlowProvider>{children}</ReactFlowProvider>
);

describe("DropoutNode", () => {

  test("renders with provided rate", () => {
    render(<DropoutNode data={{ rate: 0.25 }} />, { wrapper });
    expect(screen.getByText(/Dropout/i)).toBeInTheDocument();
  });

  test("uses fallback rate when rate not provided", () => {
    render(<DropoutNode data={{}} />, { wrapper });
    expect(screen.getByDisplayValue("0.5")).toBeInTheDocument();
  });

  test("renders without crashing when data is undefined", () => {
    render(<DropoutNode />, { wrapper });
    expect(screen.getByText("Dropout")).toBeInTheDocument();
  });

});
