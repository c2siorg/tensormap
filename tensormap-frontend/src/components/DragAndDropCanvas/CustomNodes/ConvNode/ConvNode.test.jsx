import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import ConvNode from "./ConvNode";

const deleteElementsMock = vi.fn();

vi.mock("reactflow", () => ({
  Handle: (props) => <div data-testid={`handle-${props.type}-${props.position}`} {...props} />,
  Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
  useReactFlow: () => ({ deleteElements: deleteElementsMock }),
}));

describe("ConvNode", () => {
  const defaultProps = {
    id: "test-node-conv",
    data: {
      params: {
        filter: "",
        padding: "",
        activation: "",
        kernelX: "",
        kernelY: "",
        strideX: "",
        strideY: "",
      },
    },
  };

  beforeEach(() => {
    deleteElementsMock.mockClear();
  });

  it("renders the title correctly", () => {
    render(<ConvNode {...defaultProps} />);
    expect(screen.getByText("Conv2D")).toBeInTheDocument();
  });

  it("displays 'Not configured' when parameters are unconfigured", () => {
    render(<ConvNode {...defaultProps} />);
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("renders configured state and parameters when all provided", () => {
    const configuredProps = {
      ...defaultProps,
      data: {
        params: {
          filter: 32,
          padding: "same",
          activation: "relu",
          kernelX: 3,
          kernelY: 3,
          strideX: 1,
          strideY: 1,
        },
      },
    };
    render(<ConvNode {...configuredProps} />);
    expect(screen.getByText("F: 32, K: 3×3, S: 1×1, P: same, Act: relu")).toBeInTheDocument();
  });

  it("ignores 'none' activation", () => {
    const configuredProps = {
      ...defaultProps,
      data: {
        params: {
          filter: 16,
          activation: "none",
        },
      },
    };
    render(<ConvNode {...configuredProps} />);
    expect(screen.getByText("F: 16")).toBeInTheDocument();
  });

  it("renders configured state when partial parameters are provided", () => {
    const configuredProps = {
      ...defaultProps,
      data: {
        params: {
          filter: 64,
          kernelX: 5,
        }, // kernelY missing, so K: should be missing
      },
    };
    render(<ConvNode {...configuredProps} />);
    // Since kernelY is missing, only F: 64 is rendered
    expect(screen.getByText("F: 64")).toBeInTheDocument();
  });

  it("renders correct ReactFlow handles (source and target)", () => {
    render(<ConvNode {...defaultProps} />);

    const targetHandle = screen.getByTestId("handle-target-left");
    expect(targetHandle).toBeInTheDocument();

    const sourceHandle = screen.getByTestId("handle-source-right");
    expect(sourceHandle).toBeInTheDocument();
  });

  it("renders delete button and deletes node on click", () => {
    render(<ConvNode {...defaultProps} />);

    const deleteButton = screen.getByTestId("conv-node-delete-button");
    expect(deleteButton).toBeInTheDocument();

    fireEvent.click(deleteButton);
    expect(deleteElementsMock).toHaveBeenCalledWith({ nodes: [{ id: defaultProps.id }] });
  });
});
