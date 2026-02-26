import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import InputNode from "./InputNode";

// Mock reactflow's Handle and Position since we're testing the node purely presentationally.
vi.mock("reactflow", () => ({
    Handle: (props) => <div data-testid={`handle-${props.type}-${props.position}`} {...props} />,
    Position: { Left: "left", Right: "right", Top: "top", Bottom: "bottom" },
}));

describe("InputNode", () => {
    const defaultProps = {
        id: "test-node-1",
        data: {
            params: {
                "dim-1": "",
                "dim-2": "",
                "dim-3": "",
            },
        },
    };

    it("renders the title correctly", () => {
        render(<InputNode {...defaultProps} />);
        expect(screen.getByText("Input")).toBeInTheDocument();
    });

    it("displays 'No dimensions set' when parameters are unconfigured", () => {
        render(<InputNode {...defaultProps} />);
        expect(screen.getByText("No dimensions set")).toBeInTheDocument();
    });

    it("renders configured state and parameters when dimensions are provided", () => {
        const configuredProps = {
            ...defaultProps,
            data: {
                params: {
                    "dim-1": 28,
                    "dim-2": 28,
                    "dim-3": 1,
                },
            },
        };
        render(<InputNode {...configuredProps} />);
        expect(screen.getByText("Dim: 28 × 28 × 1")).toBeInTheDocument();
    });

    it("renders configured state when partial dimensions are provided", () => {
        const configuredProps = {
            ...defaultProps,
            data: {
                params: {
                    "dim-1": 128,
                    "dim-2": "",
                    "dim-3": "",
                },
            },
        };
        render(<InputNode {...configuredProps} />);
        expect(screen.getByText("Dim: 128")).toBeInTheDocument();
    });

    it("renders the correct ReactFlow handles (source only)", () => {
        render(<InputNode {...defaultProps} />);
        // Input node should only have a source handle on the right
        const sourceHandle = screen.getByTestId("handle-source-right");
        expect(sourceHandle).toBeInTheDocument();

        // Should not have a target handle
        const targetHandleLeft = screen.queryByTestId("handle-target-left");
        expect(targetHandleLeft).not.toBeInTheDocument();
    });
});
