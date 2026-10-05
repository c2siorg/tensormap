import type { NodeAnalysisView } from "../../types/graphAnalysis";
import { formatCount, formatShape } from "../DragAndDropCanvas/analysisView";

interface NodeShapeBadgeProps {
  /** Attached to node data by the canvas from the static graph analysis. */
  analysis?: NodeAnalysisView;
}

/**
 * One line under a layer showing its inferred output shape and parameter count. Renders nothing
 * until the analysis arrives, or when the shape is unknown because the layer or something upstream
 * of it is invalid.
 */
export default function NodeShapeBadge({ analysis }: NodeShapeBadgeProps) {
  if (!analysis?.outputShape) return null;
  return (
    <div
      className="border-t border-gray-100 px-3 py-1 font-mono text-[11px] text-gray-500"
      data-testid="node-output-shape"
    >
      → {formatShape(analysis.outputShape)}
      {analysis.params > 0 && ` · ${formatCount(analysis.params)} params`}
    </div>
  );
}
