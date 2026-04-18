import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import LSTMNode from "./LSTMNode";

vi.mock("reactflow", () => ({
  Handle: (props) => <div data-testid={`handle-${props.type}-${props.position}`} {...props} />,
  Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
}));

describe("LSTMNode", () => {
  const defaultProps = {
    id: "test-node-lstm",
    data: { params: { units: "", returnSequences: "false" } },
  };

  it("renders the title correctly", () => {
    render(<LSTMNode {...defaultProps} />);
    expect(screen.getByText("LSTM")).toBeInTheDocument();
  });

  it("shows Not configured when units is empty", () => {
    render(<LSTMNode {...defaultProps} />);
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("shows units when configured", () => {
    const props = { ...defaultProps, data: { params: { units: 64, returnSequences: "false" } } };
    render(<LSTMNode {...props} />);
    expect(screen.getByText("Units: 64")).toBeInTheDocument();
  });

  it("shows seq suffix when returnSequences is true", () => {
    const props = { ...defaultProps, data: { params: { units: 32, returnSequences: "true" } } };
    render(<LSTMNode {...props} />);
    expect(screen.getByText("Units: 32 \u00b7 seq")).toBeInTheDocument();
  });

  it("renders target handle on the left", () => {
    render(<LSTMNode {...defaultProps} />);
    expect(screen.getByTestId("handle-target-left")).toBeInTheDocument();
  });

  it("renders source handle on the right", () => {
    render(<LSTMNode {...defaultProps} />);
    expect(screen.getByTestId("handle-source-right")).toBeInTheDocument();
  });

  it("shows Not configured for non-numeric units", () => {
    const props = { ...defaultProps, data: { params: { units: "abc", returnSequences: "false" } } };
    render(<LSTMNode {...props} />);
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("shows Not configured for zero units", () => {
    const props = { ...defaultProps, data: { params: { units: 0, returnSequences: "false" } } };
    render(<LSTMNode {...props} />);
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("shows Not configured for negative units", () => {
    const props = { ...defaultProps, data: { params: { units: -64, returnSequences: "false" } } };
    render(<LSTMNode {...props} />);
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("handles missing params gracefully", () => {
    const props = { id: "test", data: { params: {} } };
    expect(() => render(<LSTMNode {...props} />)).not.toThrow();
  });
});
