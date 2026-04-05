import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import DropoutNode from "./DropoutNode";

const renderNode = (params = {}) =>
  render(
    <ReactFlowProvider>
      <DropoutNode id="do-test" data={{ params: { rate: 0.5, ...params } }} />
    </ReactFlowProvider>,
  );

describe("DropoutNode", () => {
  it("renders the Dropout label", () => {
    renderNode();
    expect(screen.getByText("Dropout")).toBeDefined();
  });

  it("displays rate when set", () => {
    renderNode({ rate: 0.3 });
    expect(screen.getByText("Rate: 0.3")).toBeDefined();
  });

  it("shows Not configured when rate is empty", () => {
    renderNode({ rate: "" });
    expect(screen.getByText("Not configured")).toBeDefined();
  });
});
