import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import type { GraphAnalysis } from "../types/graphAnalysis";

export interface CanvasPayload {
  nodes: Array<{ id: string; type: string; data: { params: Record<string, unknown> } }>;
  edges: Array<{ source: string; target: string }>;
}

/**
 * Statically analyses a canvas graph on the backend: per-layer output shapes, parameter
 * counts and diagnostics naming the layer behind each problem. An invalid graph is a normal
 * result here, not an error; this only rejects on network or server failure.
 */
export const analyzeGraph = async (canvas: CanvasPayload): Promise<GraphAnalysis> => {
  const resp = await axios.post(urls.BACKEND_ANALYZE_GRAPH, { canvas });
  return resp.data as GraphAnalysis;
};
