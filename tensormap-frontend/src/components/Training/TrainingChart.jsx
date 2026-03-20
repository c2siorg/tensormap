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
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export default function TrainingChart({ epochMetrics, totalEpochs, isTraining }) {
  if (epochMetrics.length === 0 && !isTraining) return null;

  const hasVal = epochMetrics.some((d) => d.val_loss != null);
  const metricName = epochMetrics[0]?.metric_name ?? null;
  const hasMetric = epochMetrics.some((d) => d.metric != null);

  // Progress bar
  const progress = totalEpochs > 0 ? Math.round((epochMetrics.length / totalEpochs) * 100) : 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training Progress</CardTitle>
          <span className="text-xs text-muted-foreground">
            {epochMetrics.length}/{totalEpochs > 0 ? totalEpochs : "?"} epochs
          </span>
        </div>
        {/* Epoch progress bar */}
        {totalEpochs > 0 && (
          <div className="mt-1 h-1.5 w-full rounded-full bg-muted">
            <div
              className="h-1.5 rounded-full bg-primary transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Loss chart */}
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">Loss</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={epochMetrics} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="epoch"
                tick={{ fontSize: 11 }}
                label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }}
              />
              <YAxis tick={{ fontSize: 11 }} width={52} />
              <Tooltip
                formatter={(val, name) => [typeof val === "number" ? val.toFixed(4) : val, name]}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Train Loss"
                isAnimationActive={false}
              />
              {hasVal && (
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                  isAnimationActive={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Metric chart */}
        {hasMetric && (
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">
              {metricName ? metricName.charAt(0).toUpperCase() + metricName.slice(1) : "Metric"}
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={epochMetrics} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fontSize: 11 }}
                  label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }}
                />
                <YAxis tick={{ fontSize: 11 }} width={52} />
                <Tooltip
                  formatter={(val, name) => [typeof val === "number" ? val.toFixed(4) : val, name]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="metric"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  name={`Train ${metricName ?? "metric"}`}
                  isAnimationActive={false}
                />
                {hasVal && (
                  <Line
                    type="monotone"
                    dataKey="val_metric"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                    name={`Val ${metricName ?? "metric"}`}
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Final eval summary */}
        {!isTraining && epochMetrics.length > 0 && (
          <div className="flex flex-wrap gap-4 border-t pt-3 text-xs">
            {(() => {
              const last = epochMetrics[epochMetrics.length - 1];
              return (
                <>
                  <span>
                    Final loss: <strong className="text-foreground">{last.loss?.toFixed(4)}</strong>
                  </span>
                  {last.val_loss != null && (
                    <span>
                      Val loss:{" "}
                      <strong className="text-foreground">{last.val_loss?.toFixed(4)}</strong>
                    </span>
                  )}
                  {last.metric != null && (
                    <span>
                      {metricName}:{" "}
                      <strong className="text-foreground">{last.metric?.toFixed(4)}</strong>
                    </span>
                  )}
                  {last.val_metric != null && (
                    <span>
                      Val {metricName}:{" "}
                      <strong className="text-foreground">{last.val_metric?.toFixed(4)}</strong>
                    </span>
                  )}
                </>
              );
            })()}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

TrainingChart.propTypes = {
  epochMetrics: PropTypes.arrayOf(
    PropTypes.shape({
      epoch: PropTypes.number,
      loss: PropTypes.number,
      val_loss: PropTypes.number,
      metric: PropTypes.number,
      val_metric: PropTypes.number,
      metric_name: PropTypes.string,
    }),
  ).isRequired,
  totalEpochs: PropTypes.number.isRequired,
  isTraining: PropTypes.bool.isRequired,
};
