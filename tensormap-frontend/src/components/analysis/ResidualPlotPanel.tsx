/**
 * Residual Plot Panel Component
 * Scatter plot showing predicted vs actual values for regression models.
 * @module
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface ResidualPlotPanelProps {
  yPred: number[];
  yTrue: number[];
  residuals: number[];
  mae: number;
  mse: number;
  rmse?: number;
  r2?: number | null;
}

export default function ResidualPlotPanel({
  yPred,
  yTrue,
  residuals,
  mae,
  mse,
  rmse,
  r2,
}: ResidualPlotPanelProps) {
  // Prepare data for scatter plot (predicted vs actual)
  const scatterData = yPred.map((pred, idx) => ({
    predicted: pred,
    actual: yTrue[idx],
    residual: residuals[idx],
  }));

  // Find min/max for reference line
  const allValues = [...yPred, ...yTrue];
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Predicted vs Actual</CardTitle>
          <div className="flex gap-2">
            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
              MAE: {mae.toFixed(4)}
            </Badge>
            <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
              MSE: {mse.toFixed(4)}
            </Badge>
            {typeof rmse === "number" && (
              <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200">
                RMSE: {rmse.toFixed(4)}
              </Badge>
            )}
            {typeof r2 === "number" && (
              <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
                R²: {r2.toFixed(4)}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="actual"
              name="Actual"
              label={{ value: "Actual Values", position: "insideBottom", offset: -10 }}
            />
            <YAxis
              type="number"
              dataKey="predicted"
              name="Predicted"
              label={{ value: "Predicted Values", angle: -90, position: "insideLeft" }}
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              formatter={(value: number) => value.toFixed(4)}
            />
            <Scatter
              name="Predictions"
              data={scatterData}
              fill="#3b82f6"
              fillOpacity={0.6}
            />
            {/* Perfect prediction line (y=x) */}
            <ReferenceLine
              segment={[
                { x: minVal, y: minVal },
                { x: maxVal, y: maxVal },
              ]}
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: "Perfect Fit", position: "insideTopRight", fill: "#ef4444" }}
            />
          </ScatterChart>
        </ResponsiveContainer>

        <div className="mt-4 text-xs text-gray-600">
          <p>
            <strong>Red dashed line:</strong> Perfect predictions (actual = predicted)
          </p>
          <p>
            <strong>Blue dots:</strong> Model predictions. Closer to the red line indicates better performance.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
