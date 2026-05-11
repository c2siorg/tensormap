import PropTypes from "prop-types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function TrainingChart({ epochData, isTraining }) {
  if (epochData.length === 0 && !isTraining) return null;

  return (
    <div className="space-y-4">
      {/* Loss Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            Loss
            {isTraining && (
              <span className="inline-flex items-center gap-1 text-xs font-normal text-muted-foreground">
                <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                Live
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {epochData.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
              Waiting for training to start...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={epochData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "insideBottom", offset: -2 }}
                  tick={{ fontSize: 12 }}
                />
                <YAxis tick={{ fontSize: 12 }} width={55} />
                <Tooltip
                  formatter={(value) => (value != null ? value.toFixed(4) : "N/A")}
                  labelFormatter={(label) => `Epoch ${label}`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#f97316"
                  strokeWidth={2}
                  dot={false}
                  strokeDasharray="5 5"
                  name="Val Loss"
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Accuracy Chart — only show if accuracy data exists */}
      {(epochData.some((d) => d.accuracy != null) || isTraining) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            {epochData.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
                Waiting for training to start...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={epochData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="epoch"
                    label={{ value: "Epoch", position: "insideBottom", offset: -2 }}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis tick={{ fontSize: 12 }} width={55} domain={[0, 1]} />
                  <Tooltip
                    formatter={(value) => (value != null ? value.toFixed(4) : "N/A")}
                    labelFormatter={(label) => `Epoch ${label}`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                    name="Train Accuracy"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_accuracy"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="5 5"
                    name="Val Accuracy"
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

TrainingChart.propTypes = {
  epochData: PropTypes.arrayOf(
    PropTypes.shape({
      epoch: PropTypes.number,
      loss: PropTypes.number,
      val_loss: PropTypes.number,
      accuracy: PropTypes.number,
      val_accuracy: PropTypes.number,
    }),
  ).isRequired,
  isTraining: PropTypes.bool,
};
