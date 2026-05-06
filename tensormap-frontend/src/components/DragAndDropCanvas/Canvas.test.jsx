import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RecoilRoot } from "recoil";
import { MemoryRouter } from "react-router-dom";
import Canvas from "./Canvas";

// Mock ResizeObserver
window.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

vi.mock("../../services/ModelServices", () => ({
  getAllModels: vi.fn().mockResolvedValue([]),
  getModelGraph: vi.fn().mockResolvedValue({ success: false }),
  saveModel: vi.fn(),
}));

vi.mock("reactflow", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    __esModule: true,
    default: (props) => (
      <div data-testid="mock-reactflow">
        <button
          data-testid="node-custominput"
          onMouseEnter={(e) =>
            props.onNodeMouseEnter(e, { id: "1", type: "custominput", position: { x: 0, y: 0 } })
          }
          onMouseLeave={(e) => props.onNodeMouseLeave(e, { id: "1", type: "custominput" })}
          onMouseMove={(e) => props.onNodeMouseMove?.(e)}
        >
          Node 1
        </button>
        <button
          data-testid="node-unknown"
          onMouseEnter={(e) =>
            props.onNodeMouseEnter(e, { id: "2", type: "unknown", position: { x: 0, y: 0 } })
          }
          onMouseLeave={(e) => props.onNodeMouseLeave(e, { id: "2", type: "unknown" })}
        >
          Node 2
        </button>
        {props.children}
      </div>
    ),
  };
});

describe("Canvas Tooltip", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  const renderCanvas = () => {
    return render(
      <RecoilRoot>
        <MemoryRouter>
          <Canvas />
        </MemoryRouter>
      </RecoilRoot>,
    );
  };

  it("shows tooltip after 250ms delay on node hover", async () => {
    renderCanvas();
    const node = screen.getByTestId("node-custominput");

    fireEvent.mouseEnter(node, { clientX: 100, clientY: 100 });

    // The role="tooltip" might be in the DOM with opacity-0 initially
    const tooltip = screen.getByRole("tooltip");
    expect(tooltip).toHaveClass("opacity-0");

    act(() => {
      vi.advanceTimersByTime(250);
    });

    // Now it should be visible
    expect(tooltip).toHaveClass("opacity-100");
    expect(tooltip).toHaveTextContent("Defines the shape and format of the input data.");

    fireEvent.mouseLeave(node);

    // State immediately updates to opacity-0
    expect(tooltip).toHaveClass("opacity-0");
  });

  it("does not flash tooltip if mouse leaves before 250ms", async () => {
    renderCanvas();
    const node = screen.getByTestId("node-custominput");

    fireEvent.mouseEnter(node);
    const tooltip = screen.getByRole("tooltip");
    expect(tooltip).toHaveClass("opacity-0");

    act(() => {
      vi.advanceTimersByTime(100);
    });

    fireEvent.mouseLeave(node);

    act(() => {
      vi.advanceTimersByTime(200);
    });

    // Should still be hidden (timeout was cleared)
    expect(tooltip).toHaveClass("opacity-0");
  });

  it("does not show tooltip for unknown node types", async () => {
    renderCanvas();
    const node = screen.getByTestId("node-unknown");

    fireEvent.mouseEnter(node);
    const tooltip = screen.getByRole("tooltip");

    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(tooltip).toHaveClass("opacity-0");
  });
});
