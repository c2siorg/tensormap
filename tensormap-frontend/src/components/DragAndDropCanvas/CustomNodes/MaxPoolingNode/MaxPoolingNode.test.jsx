import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import MaxPoolingNode from "./MaxPoolingNode";

describe("MaxPoolingNode", () => {
  it("renders not configured when params are empty", () => {
    render(
      <ReactFlowProvider>
        <MaxPoolingNode id="1" data={{ params: { pool_size: "", stride: "", padding: "valid" } }} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Not configured")).toBeInTheDocument();
  });

  it("renders params when configured", () => {
    render(
      <ReactFlowProvider>
        <MaxPoolingNode id="1" data={{ params: { pool_size: 2, stride: 2, padding: "valid" } }} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Pool: 2 | Stride: 2 | valid")).toBeInTheDocument();
  });
});
