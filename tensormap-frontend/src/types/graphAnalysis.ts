/**
 * Types for POST /layers/analyze-graph, mirroring app/ir/analysis/models.py.
 */

export type DiagnosticSeverity = "error" | "warning";

export interface GraphDiagnostic {
  /** Stable machine-readable identifier, e.g. "rank_mismatch". */
  code: string;
  severity: DiagnosticSeverity;
  message: string;
  node_id: string | null;
  suggestion: string | null;
  related_node_ids: string[];
}

export interface NodeAnalysis {
  node_id: string;
  layer_type: string;
  input_shapes: number[][];
  /** Null when this node, or something upstream of it, is invalid. */
  output_shape: number[] | null;
  params: number;
  trainable_params: number;
  non_trainable_params: number;
  macs: number;
}

export interface GraphAnalysis {
  ok: boolean;
  complete: boolean;
  nodes: NodeAnalysis[];
  diagnostics: GraphDiagnostic[];
  output_node_ids: string[];
  total_params: number;
  trainable_params: number;
  non_trainable_params: number;
  macs: number;
  size_bytes: number;
}

/** What the canvas attaches to a ReactFlow node's data for rendering. */
export interface NodeAnalysisView {
  outputShape: number[] | null;
  params: number;
  severity: DiagnosticSeverity | null;
}
