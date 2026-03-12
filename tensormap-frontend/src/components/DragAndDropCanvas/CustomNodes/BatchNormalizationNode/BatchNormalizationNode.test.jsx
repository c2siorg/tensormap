import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import BatchNormalizationNode from "./BatchNormalizationNode";

describe("BatchNormalizationNode", () => {
  it("renders label from data", () => {
    render(
      <ReactFlowProvider>
        <BatchNormalizationNode data={{ label: "BatchNorm Test" }} />
      </ReactFlowProvider>
    );

    expect(screen.getByText("BatchNorm Test")).toBeInTheDocument();
  });

  it("falls back to default label", () => {
    render(
      <ReactFlowProvider>
        <BatchNormalizationNode data={{}} />
      </ReactFlowProvider>
    );

    expect(screen.getByText("BatchNorm")).toBeInTheDocument();
  });
});
