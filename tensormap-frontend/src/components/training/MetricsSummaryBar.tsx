/**
 * Metrics Summary Bar Component
 * Displays current epoch metrics, best epoch, and progress percentage.
 * @module
 */

import { Card, CardContent } from "@/components/ui/card";
import { MetricsSummaryBarProps } from "@/types/training";

function formatValue(value: number | undefined): string {
  if (value == null) return "N/A";
  return value.toFixed(4);
}

function formatPercent(value: number | undefined): string {
  if (value == null) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

function getMetricLabel(metric?: string | null): string {
  if (!metric) return "Val Loss:";
  const labels: Record<string, string> = {
    val_loss: "Val Loss:",
    loss: "Loss:",
    val_accuracy: "Val Accuracy:",
    accuracy: "Accuracy:",
    val_mae: "Val MAE:",
    mae: "MAE:",
    val_mse: "Val MSE:",
    mse: "MSE:",
  };
  return labels[metric] || `${metric}:`;
}

export default function MetricsSummaryBar({
  currentMetrics,
  bestEpoch,
  currentEpoch,
  totalEpochs,
}: MetricsSummaryBarProps) {
  const progressPercent = totalEpochs > 0 ? ((currentEpoch / totalEpochs) * 100).toFixed(1) : "0.0";

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Current Epoch Metrics */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-700">Current Epoch</h3>
            {currentMetrics ? (
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Loss:</span>
                  <span className="font-mono font-medium">{formatValue(currentMetrics.loss)}</span>
                </div>
                {currentMetrics.accuracy != null && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Accuracy:</span>
                    <span className="font-mono font-medium">
                      {formatPercent(currentMetrics.accuracy)}
                    </span>
                  </div>
                )}
                {currentMetrics.val_loss != null && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Val Loss:</span>
                    <span className="font-mono font-medium">
                      {formatValue(currentMetrics.val_loss)}
                    </span>
                  </div>
                )}
                {currentMetrics.val_accuracy != null && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Val Accuracy:</span>
                    <span className="font-mono font-medium">
                      {formatPercent(currentMetrics.val_accuracy)}
                    </span>
                  </div>
                )}
                {currentMetrics.lr != null && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Learning Rate:</span>
                    <span className="font-mono font-medium">
                      {currentMetrics.lr < 1e-3
                        ? currentMetrics.lr.toExponential(2)
                        : currentMetrics.lr.toFixed(4)}
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No data yet</p>
            )}
          </div>

          {/* Best Epoch */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-700">Best Epoch</h3>
            {bestEpoch ? (
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Epoch:</span>
                  <span className="font-mono font-medium">{bestEpoch.epoch}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">{getMetricLabel(bestEpoch.metric)}</span>
                  <span className="font-mono font-medium">
                    {formatValue(bestEpoch.value ?? bestEpoch.val_loss)}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No validation data</p>
            )}
          </div>

          {/* Progress */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-700">Progress</h3>
            <div className="text-sm">
              <p className="text-2xl font-bold text-blue-600">{progressPercent}%</p>
              <p className="text-gray-600 mt-1">
                {currentEpoch} of {totalEpochs} epochs
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
