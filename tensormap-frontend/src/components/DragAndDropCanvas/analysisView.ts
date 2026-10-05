import type { GraphAnalysis, DiagnosticSeverity } from "../../types/graphAnalysis";

/** Shape as the Keras-style tuple the user sees in model summaries: (28, 28, 1), (10,). */
export function formatShape(shape: number[] | null | undefined): string {
  if (!shape) return "?";
  return `(${shape.join(", ")}${shape.length === 1 ? "," : ""})`;
}

/** 1,234,567 -> "1.2M"; small counts stay exact. */
export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

const SEVERITY_CLASS: Record<DiagnosticSeverity, string> = {
  error: "analysis-error",
  warning: "analysis-warning",
};

/**
 * Attach the analysis to the ReactFlow nodes for display only. The result is never stored:
 * the saved graph and drafts keep using the plain nodes.
 *
 * Each node gets its inferred output shape and parameter count in `data.analysis`, and a
 * class that outlines it when a diagnostic points at it (an error wins over a warning).
 */
export function decorateNodes<T extends { id: string; className?: string; data?: any }>(
  nodes: T[],
  analysis: GraphAnalysis | null,
): T[] {
  if (!analysis) return nodes;

  const byNode = new Map(analysis.nodes.map((n) => [n.node_id, n]));
  const severity = new Map<string, DiagnosticSeverity>();
  for (const d of analysis.diagnostics) {
    if (!d.node_id) continue;
    if (severity.get(d.node_id) !== "error") severity.set(d.node_id, d.severity);
  }

  return nodes.map((node) => {
    const result = byNode.get(node.id);
    const sev = severity.get(node.id) ?? null;
    if (!result && !sev) return node;
    const className = [node.className, sev ? SEVERITY_CLASS[sev] : null].filter(Boolean).join(" ");
    return {
      ...node,
      className: className || undefined,
      data: {
        ...node.data,
        analysis: {
          outputShape: result?.output_shape ?? null,
          params: result?.params ?? 0,
          severity: sev,
        },
      },
    };
  });
}
