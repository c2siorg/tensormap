import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import AlphaDropoutNode from "./AlphaDropoutNode";

const renderNode = (params = {}) =>
  render(
    <ReactFlowProvider>
      <AlphaDropoutNode id="ad-test" data={{ params: { rate: "0.5", ...params } }} />
    </ReactFlowProvider>,
  );

describe("AlphaDropoutNode", () => {
  it("renders the AlphaDropout label", () => {
    renderNode();
    expect(screen.getByText("AlphaDropout")).toBeDefined();
  });

  it("displays rate when set", () => {
    renderNode({ rate: "0.3" });
    expect(screen.getByText("Rate: 0.3")).toBeDefined();
  });

  it("shows Not configured when rate is empty", () => {
    renderNode({ rate: "" });
    expect(screen.getByText("Not configured")).toBeDefined();
  });

  it("shows Not configured when rate is invalid", () => {
    renderNode({ rate: "abc" });
    expect(screen.getByText("Not configured")).toBeDefined();
  });

  it("shows Not configured when rate is 0", () => {
    renderNode({ rate: "0" });
    expect(screen.getByText("Not configured")).toBeDefined();
  });

  it("shows Not configured when rate is > 1", () => {
    renderNode({ rate: "1.5" });
    expect(screen.getByText("Not configured")).toBeDefined();
  });

  it("renders a target handle on the left side", () => {
    const { container } = renderNode();
    const left = container.querySelector("[data-handlepos='left']");
    expect(left).not.toBeNull();
  });

  it("renders a source handle on the right side", () => {
    const { container } = renderNode();
    const right = container.querySelector("[data-handlepos='right']");
    expect(right).not.toBeNull();
  });
});
