import { describe, it, expect } from "vitest";
import { decorateNodes, formatBytes, formatCount, formatShape } from "../analysisView";
import type { GraphAnalysis } from "../../../types/graphAnalysis";

const base: GraphAnalysis = {
  ok: true,
  complete: true,
  nodes: [],
  diagnostics: [],
  output_node_ids: [],
  total_params: 0,
  trainable_params: 0,
  non_trainable_params: 0,
  macs: 0,
  size_bytes: 0,
};

const node = (id: string, extra: object = {}) => ({
  id,
  position: { x: 0, y: 0 },
  data: { params: {} },
  ...extra,
});

const nodeResult = (node_id: string, output_shape: number[] | null, params = 0) => ({
  node_id,
  layer_type: "dense",
  input_shapes: [],
  output_shape,
  params,
  trainable_params: params,
  non_trainable_params: 0,
  macs: 0,
});

describe("formatShape", () => {
  it("matches the Keras tuple notation", () => {
    expect(formatShape([28, 28, 1])).toBe("(28, 28, 1)");
    expect(formatShape([10])).toBe("(10,)");
    expect(formatShape(null)).toBe("?");
  });
});

describe("formatCount / formatBytes", () => {
  it("abbreviates large counts and keeps small ones exact", () => {
    expect(formatCount(704)).toBe("704");
    expect(formatCount(12_345)).toBe("12.3K");
    expect(formatCount(1_329_473)).toBe("1.3M");
    expect(formatCount(2_500_000_000)).toBe("2.5B");
  });

  it("scales bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("decorateNodes", () => {
  it("returns the same nodes when there is no analysis", () => {
    const nodes = [node("a")];
    expect(decorateNodes(nodes, null)).toBe(nodes);
  });

  it("attaches output shape and params to each analysed node", () => {
    const [a] = decorateNodes([node("a")], { ...base, nodes: [nodeResult("a", [64], 704)] });
    expect(a.data.analysis).toEqual({ outputShape: [64], params: 704, severity: null });
    expect(a.className).toBeUndefined();
  });

  it("outlines a node a diagnostic points at, and an error wins over a warning", () => {
    const diag = (severity: "error" | "warning") => ({
      code: "x",
      severity,
      message: "m",
      node_id: "a",
      suggestion: null,
      related_node_ids: [],
    });
    const [a, b] = decorateNodes([node("a", { className: "keep" }), node("b")], {
      ...base,
      ok: false,
      nodes: [nodeResult("a", null), nodeResult("b", [4])],
      diagnostics: [diag("warning"), diag("error")],
    });
    expect(a.className).toBe("keep analysis-error");
    expect(a.data.analysis.outputShape).toBeNull();
    expect(b.className).toBeUndefined();
  });

  it("does not mutate the input nodes", () => {
    const original = node("a");
    decorateNodes([original], { ...base, nodes: [nodeResult("a", [1])] });
    expect(original.data).toEqual({ params: {} });
  });

  it("leaves nodes the analysis does not mention untouched", () => {
    const n = node("ghost");
    expect(decorateNodes([n], base)[0]).toBe(n);
  });
});
