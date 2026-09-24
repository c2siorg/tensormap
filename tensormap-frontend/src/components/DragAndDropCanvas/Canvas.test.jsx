import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RecoilRoot } from "recoil";
import { MemoryRouter } from "react-router-dom";
import Canvas from "./Canvas";
import {
  getMiniMapNodeColor,
  MINIMAP_NODE_COLORS,
  MINIMAP_FALLBACK_COLOR,
} from "../../constants/nodeColors";

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

// Only `merge` matters to the connection guard; Concatenate is the one layer
// the backend registry flags as accepting multiple inputs.
vi.mock("../../hooks/useLayerRegistry", async (importOriginal) => {
  const actual = await importOriginal();
  const specs = {
    input: { type_key: "input", display_name: "Input", merge: false, params: {} },
    dense: { type_key: "dense", display_name: "Dense", merge: false, params: {} },
    concatenate: { type_key: "concatenate", display_name: "Concatenate", merge: true, params: {} },
  };
  return { ...actual, getLayerSpec: (typeKey) => specs[typeKey] ?? null };
});

// Latest props the canvas handed to <ReactFlow>, so tests can drive the
// connection callbacks the way the library would.
const reactFlowProps = vi.hoisted(() => ({ current: null }));

vi.mock("reactflow", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    __esModule: true,
    // The real MiniMap measures the canvas through the ReactFlow store, which
    // jsdom cannot provide; record the props the canvas passes instead.
    MiniMap: (props) => (
      <div
        data-testid="rf__minimap"
        className={`react-flow__minimap ${props.className ?? ""}`}
        data-position={props.position}
        data-pannable={String(!!props.pannable)}
        data-zoomable={String(!!props.zoomable)}
        aria-label={props.ariaLabel}
      />
    ),
    default: (props) => {
      reactFlowProps.current = props;
      return renderMockReactFlow(props);
    },
  };
});

const renderMockReactFlow = (props) => (
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
);

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

describe("Canvas MiniMap", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders a pannable, zoomable minimap on an empty canvas", () => {
    render(
      <RecoilRoot>
        <MemoryRouter>
          <Canvas />
        </MemoryRouter>
      </RecoilRoot>,
    );

    const minimap = screen.getByTestId("rf__minimap");
    expect(minimap).toBeInTheDocument();
    expect(minimap).toHaveAttribute("data-position", "bottom-right");
    expect(minimap).toHaveAttribute("data-pannable", "true");
    expect(minimap).toHaveAttribute("data-zoomable", "true");
  });

  it("hides the minimap when the canvas is too narrow to fit it beside the controls", () => {
    render(
      <RecoilRoot>
        <MemoryRouter>
          <Canvas />
        </MemoryRouter>
      </RecoilRoot>,
    );

    expect(screen.getByTestId("rf__minimap")).toHaveClass("!hidden", "xl:!block");
  });

  it("returns a distinct color for every bespoke node type", () => {
    const types = Object.keys(MINIMAP_NODE_COLORS);
    expect(types).toHaveLength(7);

    const colors = types.map((type) => getMiniMapNodeColor({ type }));
    expect(new Set(colors).size).toBe(types.length);
    colors.forEach((color) => expect(color).not.toBe(MINIMAP_FALLBACK_COLOR));
  });

  it("falls back to a neutral color for registry-driven and unknown node types", () => {
    expect(getMiniMapNodeColor({ type: "genericlayer" })).toBe(MINIMAP_FALLBACK_COLOR);
    expect(getMiniMapNodeColor({ type: "lstm" })).toBe(MINIMAP_FALLBACK_COLOR);
    expect(getMiniMapNodeColor({})).toBe(MINIMAP_FALLBACK_COLOR);
  });
});

describe("Canvas connection guard", () => {
  // Canvas restores this draft on mount (no projectId in MemoryRouter).
  const DRAFT_KEY = "tensormap_draft_default";
  const layer = (id, layerType) => ({
    id,
    type: "genericlayer",
    position: { x: 0, y: 0 },
    data: { layerType, params: {}, label: layerType },
  });
  const draft = {
    modelName: "",
    nodes: [
      layer("n1", "input"),
      layer("n2", "dense"),
      layer("n3", "dense"),
      layer("n4", "concatenate"),
    ],
    edges: [
      { id: "e1", source: "n1", target: "n2" },
      { id: "e2", source: "n1", target: "n4", targetHandle: "input-0" },
    ],
  };
  const connection = (source, target, targetHandle = null) => ({
    source,
    sourceHandle: null,
    target,
    targetHandle,
  });

  const renderCanvas = () => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
    return render(
      <RecoilRoot>
        <MemoryRouter>
          <Canvas />
        </MemoryRouter>
      </RecoilRoot>,
    );
  };

  const attempt = (conn) => {
    let result;
    act(() => {
      result = reactFlowProps.current.isValidConnection(conn);
    });
    return result;
  };

  afterEach(() => {
    localStorage.removeItem(DRAFT_KEY);
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("refuses a self-loop", () => {
    renderCanvas();
    expect(attempt(connection("n2", "n2"))).toBe(false);
    expect(screen.getByRole("alert")).toHaveTextContent("A layer cannot connect to itself");
  });

  it("refuses a duplicate edge", () => {
    renderCanvas();
    expect(attempt(connection("n1", "n2"))).toBe(false);
    expect(screen.getByRole("alert")).toHaveTextContent("These layers are already connected");
  });

  it("refuses a second input on a single-input layer", () => {
    renderCanvas();
    expect(attempt(connection("n3", "n2"))).toBe(false);
    expect(screen.getByRole("alert")).toHaveTextContent("Dense accepts only one input");
  });

  it("allows a second input on a merge layer", () => {
    renderCanvas();
    expect(attempt(connection("n3", "n4", "input-1"))).toBe(true);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("accepts a valid connection, adds the edge and pushes an undo snapshot", () => {
    renderCanvas();
    const undoButton = screen.getByTitle(/^Undo/);
    expect(undoButton).toBeDisabled();
    expect(reactFlowProps.current.edges).toHaveLength(2);

    const conn = connection("n2", "n3");
    expect(attempt(conn)).toBe(true);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    act(() => {
      reactFlowProps.current.onConnect(conn);
    });

    expect(reactFlowProps.current.edges).toHaveLength(3);
    expect(reactFlowProps.current.edges[2]).toMatchObject({ source: "n2", target: "n3" });
    expect(undoButton).toBeEnabled();
  });

  it("dismisses the notice after 3 seconds", () => {
    vi.useFakeTimers();
    renderCanvas();
    attempt(connection("n2", "n2"));
    expect(screen.getByRole("alert")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2999);
    });
    expect(screen.getByRole("alert")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("clears the notice as soon as a valid connection is made", () => {
    renderCanvas();
    attempt(connection("n3", "n2"));
    expect(screen.getByRole("alert")).toBeInTheDocument();

    act(() => {
      reactFlowProps.current.onConnect(connection("n2", "n3"));
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
