import { useEffect, useMemo, useState } from "react";
import { generateModelJSON } from "../components/DragAndDropCanvas/Helpers";
import { analyzeGraph, type CanvasPayload } from "../services/graphAnalysisService";
import type { GraphAnalysis } from "../types/graphAnalysis";
import logger from "../shared/logger";

/** Wait for editing to pause before asking the backend, so typing in a field sends one request. */
export const ANALYSIS_DEBOUNCE_MS = 400;

/**
 * Reduce a canvas to what affects the analysis. Node positions are dropped so that dragging a
 * layer around does not trigger a new request.
 */
export function toAnalysisPayload(nodes: any[], edges: any[]): CanvasPayload {
  const { nodes: modelNodes, edges: modelEdges } = generateModelJSON({ nodes, edges });
  return {
    nodes: modelNodes.map(({ id, type, data }) => ({ id, type, data })),
    edges: modelEdges,
  };
}

interface UseGraphAnalysisResult {
  /** The latest analysis, or null when the canvas is empty or the request failed. */
  analysis: GraphAnalysis | null;
  /** True from an edit until its analysis arrives. */
  loading: boolean;
  /** Set when the backend could not be reached; the canvas keeps working without analysis. */
  error: string | null;
  /** True when displaying an earlier analysis because the latest request failed. */
  isStale: boolean;
}

/**
 * Keeps a live static analysis of the canvas: output shapes, parameter counts and diagnostics.
 *
 * Requests are debounced, stale responses are discarded, and a failure never blocks editing.
 */
export function useGraphAnalysis(
  nodes: any[],
  edges: any[],
  debounceMs: number = ANALYSIS_DEBOUNCE_MS,
): UseGraphAnalysisResult {
  const [analysis, setAnalysis] = useState<GraphAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isStale, setIsStale] = useState(false);

  const hasNodes = nodes.length > 0;
  // A string key, so the effect re-runs only when the graph really changes.
  const key = useMemo(
    () => (hasNodes ? JSON.stringify(toAnalysisPayload(nodes, edges)) : ""),
    [hasNodes, nodes, edges],
  );

  useEffect(() => {
    if (!key) {
      setAnalysis(null);
      setLoading(false);
      setError(null);
      setIsStale(false);
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const result = await analyzeGraph(JSON.parse(key));
        if (cancelled) return;
        setAnalysis(result);
        setError(null);
        setIsStale(false);
      } catch (err) {
        if (cancelled) return;
        logger.error("Graph analysis failed:", err);
        // Preserve existing analysis if available so shape badges and stats remain visible
        setError("Could not analyse the model.");
        setIsStale(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, debounceMs);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, debounceMs]);

  return { analysis, loading, error, isStale };
}
