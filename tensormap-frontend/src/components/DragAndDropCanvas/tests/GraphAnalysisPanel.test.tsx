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
});
