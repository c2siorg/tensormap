import { useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { GraphAnalysis } from "../../types/graphAnalysis";
import { formatBytes, formatCount } from "./analysisView";

interface GraphAnalysisPanelProps {
  analysis: GraphAnalysis | null;
  loading?: boolean;
  error?: string | null;
  /** Called with a node id when the user picks a diagnostic, so the canvas can focus that layer. */
  onSelectNode?: (nodeId: string) => void;
}

/**
 * Live model check: totals for the architecture drawn so far, and every problem the static
 * analysis found, each linked to the layer that causes it.
 */
export default function GraphAnalysisPanel({
  analysis,
  loading = false,
  error = null,
  onSelectNode,
}: GraphAnalysisPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (!analysis && !error) return null;

  const errors = analysis?.diagnostics.filter((d) => d.severity === "error") ?? [];
  const warnings = analysis?.diagnostics.filter((d) => d.severity === "warning") ?? [];

  let status: string;
  if (error) status = error;
  else if (errors.length) status = `${errors.length} ${errors.length === 1 ? "error" : "errors"}`;
  else if (warnings.length)
    status = `${warnings.length} ${warnings.length === 1 ? "warning" : "warnings"}`;
  else status = "No problems found";

  return (
    <Card className="mt-2" aria-busy={loading}>
      <CardHeader className="flex flex-row items-center justify-between py-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          {errors.length ? (
            <AlertCircle className="h-4 w-4 text-red-600" aria-hidden />
          ) : warnings.length ? (
            <AlertTriangle className="h-4 w-4 text-amber-600" aria-hidden />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-green-600" aria-hidden />
          )}
          Model check
          <span className="font-normal text-muted-foreground" data-testid="analysis-status">
            {status}
          </span>
        </CardTitle>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand model check" : "Collapse model check"}
        >
          {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
        </Button>
      </CardHeader>

      {!collapsed && analysis && (
        <CardContent className="space-y-2 pt-0 text-xs">
          {analysis.total_params > 0 && (
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted-foreground">
              <span>
                Parameters:{" "}
                <strong className="text-foreground">{formatCount(analysis.total_params)}</strong>
              </span>
              {analysis.non_trainable_params > 0 && (
                <span>
                  Non-trainable:{" "}
                  <strong className="text-foreground">
                    {formatCount(analysis.non_trainable_params)}
                  </strong>
                </span>
              )}
              <span>
                Size:{" "}
                <strong className="text-foreground">{formatBytes(analysis.size_bytes)}</strong>
              </span>
              <span>
                MACs per sample:{" "}
                <strong className="text-foreground">{formatCount(analysis.macs)}</strong>
              </span>
            </div>
          )}

          {analysis.diagnostics.length > 0 && (
            <ul className="space-y-1.5">
              {[...errors, ...warnings].map((d, i) => (
                <li
                  key={`${d.code}-${d.node_id ?? "graph"}-${i}`}
                  className={`rounded border px-2 py-1.5 ${
                    d.severity === "error"
                      ? "border-red-200 bg-red-50"
                      : "border-amber-200 bg-amber-50"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span>{d.message}</span>
                    {d.node_id && onSelectNode && (
                      <Button
                        variant="link"
                        size="sm"
                        className="h-auto shrink-0 p-0 text-xs"
                        onClick={() => onSelectNode(d.node_id as string)}
                      >
                        Show layer
                      </Button>
                    )}
                  </div>
                  {d.suggestion && (
                    <div className="mt-0.5 text-muted-foreground">{d.suggestion}</div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      )}
    </Card>
  );
}
