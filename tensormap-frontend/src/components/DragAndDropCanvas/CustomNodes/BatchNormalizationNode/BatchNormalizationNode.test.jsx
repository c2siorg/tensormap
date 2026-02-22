import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import BatchNormalizationNode from "./BatchNormalizationNode";

const renderNode = (params = {}) =>
  render(
    <ReactFlowProvider>
      <BatchNormalizationNode
        id="bn-test"
        data={{ params: { momentum: 0.99, epsilon: 0.001, ...params } }}
      />
    </ReactFlowProvider>,
  );

describe("BatchNormalizationNode", () => {
  it("renders the BatchNorm label", () => {
    renderNode();
    expect(screen.getByText("BatchNorm")).toBeDefined();
  });

  it("displays momentum and epsilon when set", () => {
    renderNode({ momentum: 0.99, epsilon: 0.001 });
    expect(screen.getByText(/Mom: 0.99/)).toBeDefined();
  });

  it("shows Not configured when params empty", () => {
    renderNode({ momentum: "", epsilon: "" });
    expect(screen.getByText("Not configured")).toBeDefined();
  });
});
