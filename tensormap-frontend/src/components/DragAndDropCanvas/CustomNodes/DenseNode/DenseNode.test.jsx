import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import DenseNode from "./DenseNode";

const deleteElementsMock = vi.fn();

vi.mock("reactflow", () => ({
  Handle: (props) => <div data-testid={`handle-${props.type}-${props.position}`} {...props} />,
  Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
  useReactFlow: () => ({ deleteElements: deleteElementsMock }),
}));

describe("DenseNode", () => {
  const defaultProps = {
    id: "test-node-dense",
    data: {
      params: {
        units: "",
        activation: "",
      },
    },
  };

  beforeEach(() => {
    deleteElementsMock.mockClear();
  });

  it("renders the title correctly", () => {
    render(<DenseNode {...defaultProps} />);
    expect(screen.getByText("Dense")).toBeInTheDocument();
  });

  it("displays 'Not configured' when parameters are unconfigured", () => {
    render(<DenseNode {...defaultProps} />);
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("renders configured state and parameters when units and activation are provided", () => {
    const configuredProps = {
      ...defaultProps,
      data: {
        params: {
          units: 128,
          activation: "relu",
        },
      },
    };
    render(<DenseNode {...configuredProps} />);
    expect(screen.getByText("Units: 128, Act: relu")).toBeInTheDocument();
  });

  it("renders configured state when only units are provided", () => {
    const configuredProps = {
      ...defaultProps,
      data: {
        params: {
          units: 64,
          activation: "",
        },
      },
    };
    render(<DenseNode {...configuredProps} />);
    expect(screen.getByText("Units: 64")).toBeInTheDocument();
  });

  it("renders correct ReactFlow handles (source and target)", () => {
    render(<DenseNode {...defaultProps} />);

    // Dense node should have a target handle on the left
    const targetHandle = screen.getByTestId("handle-target-left");
    expect(targetHandle).toBeInTheDocument();

    // and a source handle on the right
    const sourceHandle = screen.getByTestId("handle-source-right");
    expect(sourceHandle).toBeInTheDocument();
  });

  it("renders delete button and deletes node on click", () => {
    render(<DenseNode {...defaultProps} />);

    const deleteButton = screen.getByRole("button", { name: "Delete layer" });
    expect(deleteButton).toBeInTheDocument();
    expect(deleteButton).toHaveAttribute("title", "Delete layer");

    fireEvent.click(deleteButton);
    expect(deleteElementsMock).toHaveBeenCalledWith({ nodes: [{ id: defaultProps.id }] });
  });
});
