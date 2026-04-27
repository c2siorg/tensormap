import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import FlattenNode from "./FlattenNode";

const deleteElementsMock = vi.fn();

vi.mock("reactflow", () => ({
  Handle: (props) => <div data-testid={`handle-${props.type}-${props.position}`} {...props} />,
  Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
  useReactFlow: () => ({ deleteElements: deleteElementsMock }),
}));

describe("FlattenNode", () => {
  const defaultProps = {
    id: "test-node-flatten",
  };

  beforeEach(() => {
    deleteElementsMock.mockClear();
  });

  it("renders the title correctly", () => {
    render(<FlattenNode {...defaultProps} />);
    expect(screen.getByText("Flatten")).toBeInTheDocument();
  });

  it("displays 'No parameters' correctly", () => {
    render(<FlattenNode {...defaultProps} />);
    expect(screen.getByText("No parameters")).toBeInTheDocument();
  });

  it("renders correct ReactFlow handles (source and target)", () => {
    render(<FlattenNode {...defaultProps} />);

    // Flatten node should have a target handle on the left
    const targetHandle = screen.getByTestId("handle-target-left");
    expect(targetHandle).toBeInTheDocument();

    // and a source handle on the right
    const sourceHandle = screen.getByTestId("handle-source-right");
    expect(sourceHandle).toBeInTheDocument();
  });

  it("renders delete button and deletes node on click", () => {
    render(<FlattenNode {...defaultProps} />);

    const deleteButton = screen.getByRole("button", { name: "Delete layer" });
    expect(deleteButton).toBeInTheDocument();
    expect(deleteButton).toHaveAttribute("title", "Delete layer");

    fireEvent.click(deleteButton);
    expect(deleteElementsMock).toHaveBeenCalledWith({ nodes: [{ id: defaultProps.id }] });
  });
});
