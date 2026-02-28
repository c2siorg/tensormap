import { useState } from "react";
import PropTypes from "prop-types";
import { ChevronDown, ChevronUp, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function ModelSummaryPanel({ summary, onClose }) {
  const [collapsed, setCollapsed] = useState(false);

  if (!summary) return null;

  return (
    <Card className="mt-4">
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <CardTitle className="text-sm">Model Architecture Summary</CardTitle>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => setCollapsed((c) => !c)}
            aria-label={collapsed ? "Expand summary" : "Collapse summary"}
          >
            {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={onClose}
            aria-label="Close summary"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      {!collapsed && (
        <CardContent className="pt-0">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b">
                  <th className="py-2 text-left font-medium">Layer</th>
                  <th className="py-2 text-left font-medium">Type</th>
                  <th className="py-2 text-left font-medium">Output Shape</th>
                  <th className="py-2 text-right font-medium">Params</th>
                </tr>
              </thead>
              <tbody>
                {summary.layers.map((layer, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-1.5 pr-3 font-mono">{layer.name}</td>
                    <td className="py-1.5 pr-3">{layer.type}</td>
                    <td className="py-1.5 pr-3 font-mono text-muted-foreground">
                      {layer.output_shape}
                    </td>
                    <td className="py-1.5 text-right tabular-nums">
                      {layer.param_count.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex flex-wrap gap-4 border-t pt-3 text-xs text-muted-foreground">
            <span>
              Total params:{" "}
              <strong className="text-foreground">{summary.total_params.toLocaleString()}</strong>
            </span>
            <span>
              Trainable:{" "}
              <strong className="text-foreground">
                {summary.trainable_params.toLocaleString()}
              </strong>
            </span>
            <span>
              Non-trainable:{" "}
              <strong className="text-foreground">
                {summary.non_trainable_params.toLocaleString()}
              </strong>
            </span>
          </div>
        </CardContent>
      )}
    </Card>
  );
}

ModelSummaryPanel.propTypes = {
  summary: PropTypes.shape({
    layers: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        type: PropTypes.string.isRequired,
        output_shape: PropTypes.string.isRequired,
        param_count: PropTypes.number.isRequired,
      }),
    ).isRequired,
    total_params: PropTypes.number.isRequired,
    trainable_params: PropTypes.number.isRequired,
    non_trainable_params: PropTypes.number.isRequired,
  }),
  onClose: PropTypes.func.isRequired,
};
