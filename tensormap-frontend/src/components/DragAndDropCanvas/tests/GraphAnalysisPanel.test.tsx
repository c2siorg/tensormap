import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import GraphAnalysisPanel from "../GraphAnalysisPanel";
import type { GraphAnalysis } from "../../../types/graphAnalysis";

const analysis = (overrides: Partial<GraphAnalysis> = {}): GraphAnalysis => ({
  ok: true,
  complete: true,
  nodes: [],
  diagnostics: [],
  output_node_ids: [],
  total_params: 2883,
  trainable_params: 2883,
  non_trainable_params: 0,
  macs: 2800,
  size_bytes: 11532,
  ...overrides,
});

describe("GraphAnalysisPanel", () => {
  it("renders nothing without an analysis or an error", () => {
    const { container } = render(<GraphAnalysisPanel analysis={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("summarises a valid model", () => {
    render(<GraphAnalysisPanel analysis={analysis()} />);
    expect(screen.getByTestId("analysis-status")).toHaveTextContent("No problems found");
    expect(screen.getByText("2,883")).toBeInTheDocument();
    expect(screen.getByText("11.3 KB")).toBeInTheDocument();
  });

  it("lists errors before warnings with their suggestions, and focuses the layer on request", () => {
    const onSelectNode = vi.fn();
    render(
      <GraphAnalysisPanel
        onSelectNode={onSelectNode}
        analysis={analysis({
          ok: false,
          diagnostics: [
            {
              code: "unflattened_dense",
              severity: "warning",
              message: "Dense 'd' keeps a 3-D output.",
              node_id: "d",
              suggestion: "Add a Flatten layer.",
              related_node_ids: [],
            },
            {
              code: "rank_mismatch",
              severity: "error",
              message: "Conv2D 'c' needs a 3-D input.",
              node_id: "c",
              suggestion: "Insert a Reshape layer.",
              related_node_ids: ["in"],
            },
          ],
        })}
      />,
    );

    expect(screen.getByTestId("analysis-status")).toHaveTextContent("1 error");
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Conv2D 'c' needs a 3-D input.");
    expect(items[0]).toHaveTextContent("Insert a Reshape layer.");
    expect(items[1]).toHaveTextContent("Dense 'd' keeps a 3-D output.");

    fireEvent.click(screen.getAllByRole("button", { name: "Show layer" })[0]);
    expect(onSelectNode).toHaveBeenCalledWith("c");

    fireEvent.click(screen.getByRole("button", { name: "Related (in)" }));
    expect(onSelectNode).toHaveBeenCalledWith("in");
  });

  it("shows cached status when displaying preserved analysis after a network error", () => {
    render(
      <GraphAnalysisPanel
        analysis={analysis()}
        error="Could not analyse the model."
        isStale={true}
      />,
    );
    expect(screen.getByTestId("analysis-status")).toHaveTextContent(
      "Could not analyse the model. (cached)",
    );
    expect(screen.getByText("2,883")).toBeInTheDocument();
  });

  it("offers no layer link for a graph-level problem", () => {
    render(
      <GraphAnalysisPanel
        onSelectNode={vi.fn()}
        analysis={analysis({
          ok: false,
          diagnostics: [
            {
              code: "no_input_node",
              severity: "error",
              message: "The graph has no Input layer.",
              node_id: null,
              suggestion: null,
              related_node_ids: [],
            },
          ],
        })}
      />,
    );
    expect(screen.queryByRole("button", { name: "Show layer" })).not.toBeInTheDocument();
  });

  it("shows the failure when the backend could not be reached", () => {
    render(<GraphAnalysisPanel analysis={null} error="Could not analyse the model." />);
    expect(screen.getByTestId("analysis-status")).toHaveTextContent("Could not analyse the model.");
  });

  it("collapses", () => {
    render(<GraphAnalysisPanel analysis={analysis()} />);
    fireEvent.click(screen.getByRole("button", { name: "Collapse model check" }));
    expect(screen.queryByText("2,883")).not.toBeInTheDocument();
  });

  it("displays summary metrics for parameterless graphs", () => {
    render(
      <GraphAnalysisPanel
        analysis={analysis({
          total_params: 0,
          trainable_params: 0,
          non_trainable_params: 0,
          size_bytes: 0,
          macs: 0,
          flops: 0,
          activation_bytes: 3136,
          nodes: [
            {
              node_id: "in",
              layer_type: "input",
              input_shapes: [],
              output_shape: [784],
              params: 0,
              trainable_params: 0,
              non_trainable_params: 0,
              macs: 0,
            },
          ],
        })}
      />,
    );
    expect(screen.getByText("Parameters:")).toBeInTheDocument();
    expect(screen.getAllByText("0")).toHaveLength(2);
    expect(screen.getByText("3.1 KB")).toBeInTheDocument();
  });

  it("renders FLOPs and activation memory when present", () => {
    render(
      <GraphAnalysisPanel
        analysis={analysis({
          macs: 5000,
          flops: 10000,
          activation_bytes: 4096,
        })}
      />,
    );
    expect(screen.getByText("FLOPs:")).toBeInTheDocument();
    expect(screen.getByText("10.0K")).toBeInTheDocument();
    expect(screen.getByText("Act. memory:")).toBeInTheDocument();
    expect(screen.getByText("4.0 KB")).toBeInTheDocument();
  });

  it("deduplicates related node chips", () => {
    render(
      <GraphAnalysisPanel
        onSelectNode={vi.fn()}
        analysis={analysis({
          ok: false,
          diagnostics: [
            {
              code: "cycle",
              severity: "error",
              message: "Cycle detected.",
              node_id: "n1",
              suggestion: "Remove cyclic edge.",
              related_node_ids: ["n2", "n2", "n3"],
            },
          ],
        })}
      />,
    );
    const relatedButtons = screen.getAllByRole("button", { name: /Related/ });
    expect(relatedButtons).toHaveLength(2);
  });
});
