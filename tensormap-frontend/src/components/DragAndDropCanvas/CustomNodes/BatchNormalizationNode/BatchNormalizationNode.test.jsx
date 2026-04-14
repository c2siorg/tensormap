import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import BatchNormalizationNode from "./BatchNormalizationNode";

describe("BatchNormalizationNode", () => {
  it("renders not configured when params are empty", () => {
    render(
      <ReactFlowProvider>
        <BatchNormalizationNode id="1" data={{ params: { momentum: "", epsilon: "" } }} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("renders params when configured", () => {
    render(
      <ReactFlowProvider>
        <BatchNormalizationNode id="1" data={{ params: { momentum: 0.99, epsilon: 0.001 } }} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Momentum: 0.99 | Epsilon: 0.001")).toBeInTheDocument();
  });
});
