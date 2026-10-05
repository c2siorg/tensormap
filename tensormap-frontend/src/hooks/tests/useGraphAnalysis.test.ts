import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useGraphAnalysis, toAnalysisPayload, ANALYSIS_DEBOUNCE_MS } from "../useGraphAnalysis";
import { analyzeGraph } from "../../services/graphAnalysisService";

vi.mock("../../services/graphAnalysisService", () => ({ analyzeGraph: vi.fn() }));

const result = (total_params: number) => ({
  ok: true,
  complete: true,
  nodes: [],
  diagnostics: [],
  output_node_ids: [],
  total_params,
  trainable_params: total_params,
  non_trainable_params: 0,
  macs: 0,
  size_bytes: 0,
});

const inputNode = (x = 0) => ({
  id: "in",
  type: "genericlayer",
  position: { x, y: 0 },
  data: { layerType: "input", params: { shape: 10 } },
});
const denseNode = {
  id: "d",
  type: "genericlayer",
  position: { x: 0, y: 100 },
  data: { layerType: "dense", params: { units: 4 } },
};
const edge = { id: "e", source: "in", target: "d" };

const flush = async (ms = ANALYSIS_DEBOUNCE_MS) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};

describe("toAnalysisPayload", () => {
  it("sends layer types and params but not positions", () => {
    const payload = toAnalysisPayload([inputNode(), denseNode], [edge]);
    expect(payload).toEqual({
      nodes: [
        { id: "in", type: "input", data: { params: { shape: 10 } } },
        { id: "d", type: "dense", data: { params: { units: 4 } } },
      ],
      edges: [{ source: "in", target: "d" }],
    });
  });
});

describe("useGraphAnalysis", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(analyzeGraph).mockReset();
    vi.mocked(analyzeGraph).mockResolvedValue(result(44));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("does nothing for an empty canvas", async () => {
    const { result: hook } = renderHook(() => useGraphAnalysis([], []));
    await flush();
    expect(analyzeGraph).not.toHaveBeenCalled();
    expect(hook.current).toEqual({ analysis: null, loading: false, error: null });
  });

  it("analyses after the debounce and exposes the result", async () => {
    const nodes = [inputNode(), denseNode];
    const { result: hook } = renderHook(() => useGraphAnalysis(nodes, [edge]));
    expect(hook.current.loading).toBe(true);
    expect(analyzeGraph).not.toHaveBeenCalled();

    await flush();

    expect(analyzeGraph).toHaveBeenCalledTimes(1);
    expect(hook.current.analysis?.total_params).toBe(44);
    expect(hook.current.loading).toBe(false);
  });

  it("sends one request for a burst of edits", async () => {
    const { rerender } = renderHook(({ nodes }) => useGraphAnalysis(nodes, [edge]), {
      initialProps: { nodes: [inputNode(), denseNode] },
    });
    for (const units of [5, 6, 7]) {
      rerender({
        nodes: [inputNode(), { ...denseNode, data: { ...denseNode.data, params: { units } } }],
      });
      await flush(100);
    }
    await flush();
    expect(analyzeGraph).toHaveBeenCalledTimes(1);
  });

  it("does not re-analyse when a node is only dragged", async () => {
    const { rerender } = renderHook(
      ({ x }) => useGraphAnalysis([inputNode(x), denseNode], [edge]),
      {
        initialProps: { x: 0 },
      },
    );
    await flush();
    rerender({ x: 250 });
    await flush();
    expect(analyzeGraph).toHaveBeenCalledTimes(1);
  });

  it("ignores a response that arrives after a newer edit", async () => {
    let resolveFirst: (v: ReturnType<typeof result>) => void = () => {};
    vi.mocked(analyzeGraph)
      .mockImplementationOnce(() => new Promise((r) => (resolveFirst = r)))
      .mockResolvedValueOnce(result(99));

    const { result: hook, rerender } = renderHook(
      ({ units }) =>
        useGraphAnalysis(
          [inputNode(), { ...denseNode, data: { ...denseNode.data, params: { units } } }],
          [edge],
        ),
      { initialProps: { units: 4 } },
    );
    await flush(); // First request is now in flight.
    rerender({ units: 8 });
    await flush(); // Second request resolves with 99.
    await act(async () => resolveFirst(result(1))); // The stale one lands last.

    expect(hook.current.analysis?.total_params).toBe(99);
  });

  it("reports a failure without throwing and recovers on the next edit", async () => {
    vi.mocked(analyzeGraph).mockRejectedValueOnce(new Error("network"));
    const { result: hook, rerender } = renderHook(
      ({ units }) =>
        useGraphAnalysis(
          [inputNode(), { ...denseNode, data: { ...denseNode.data, params: { units } } }],
          [edge],
        ),
      { initialProps: { units: 4 } },
    );
    await flush();
    expect(hook.current.error).toBe("Could not analyse the model.");
    expect(hook.current.analysis).toBeNull();

    rerender({ units: 5 });
    await flush();
    expect(hook.current.error).toBeNull();
    expect(hook.current.analysis?.total_params).toBe(44);
  });

  it("clears the analysis when the canvas is emptied", async () => {
    const { result: hook, rerender } = renderHook(({ nodes }) => useGraphAnalysis(nodes, []), {
      initialProps: { nodes: [inputNode()] as any[] },
    });
    await flush();
    expect(hook.current.analysis).not.toBeNull();
    rerender({ nodes: [] });
    expect(hook.current.analysis).toBeNull();
  });
});
