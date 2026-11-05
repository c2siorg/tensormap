/**
 * Training Metrics Chart Component
 * Main component that orchestrates all training visualization components.
 * @module
 */

import { useMemo, useEffect } from "react";
import { useTrainingMetrics } from "@/hooks/useTrainingMetrics";
import TrainingHeader from "./TrainingHeader";
import LossChart from "./LossChart";
import AccuracyChart from "./AccuracyChart";
import MetricsSummaryBar from "./MetricsSummaryBar";
import { TrainingMetricsChartProps } from "@/types/training";

export default function TrainingMetricsChart({
  jobId,
  totalEpochs,
  onComplete,
}: TrainingMetricsChartProps) {
  const { metrics, status, currentEpoch, cancel, cancelRequested, startedAt, error, isCompleted, bestCheckpoint } =
    useTrainingMetrics(jobId, totalEpochs);

  // Call onComplete when training finishes
  useEffect(() => {
    if (isCompleted && onComplete) {
      onComplete(jobId);
    }
  }, [isCompleted, jobId, onComplete]);

  // Calculate elapsed time
  const elapsedTime = useMemo(() => {
    if (!startedAt) return 0;
    const start = new Date(startedAt).getTime();
    const now = Date.now();
    return Math.floor((now - start) / 1000);
  }, [startedAt, currentEpoch]); // Re-calculate when epoch changes

  // Calculate ETA
  const eta = useMemo(() => {
    if (currentEpoch === 0 || elapsedTime === 0) return 0;
    const timePerEpoch = elapsedTime / currentEpoch;
    const remainingEpochs = totalEpochs - currentEpoch;
    return Math.floor(timePerEpoch * remainingEpochs);
  }, [currentEpoch, elapsedTime, totalEpochs]);

  // Get current metrics (last epoch)
  const currentMetrics = useMemo(() => {
    return metrics.length > 0 ? metrics[metrics.length - 1] : null;
  }, [metrics]);

  // Find best epoch (optimal checkpoint or lowest val_loss)
  const bestEpoch = useMemo(() => {
    if (bestCheckpoint) {
      return { epoch: bestCheckpoint.epoch, val_loss: bestCheckpoint.value };
    }
    const withValidation = metrics.filter((m) => m.val_loss != null);
    if (withValidation.length === 0) return null;

    const best = withValidation.reduce((prev, curr) =>
      curr.val_loss! < prev.val_loss! ? curr : prev,
    );

    return { epoch: best.epoch, val_loss: best.val_loss! };
  }, [bestCheckpoint, metrics]);

  return (
    <div className="space-y-6">
      {/* Training Header */}
      <TrainingHeader
        jobId={jobId}
        status={status}
        currentEpoch={currentEpoch}
        totalEpochs={totalEpochs}
        elapsedTime={elapsedTime}
        eta={eta}
        onStop={cancel}
        cancelRequested={cancelRequested}
      />

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">
            <strong>Error:</strong> {error}
          </p>
        </div>
      )}

      {/* Charts Grid */}
      {metrics.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <LossChart metrics={metrics} />
            <AccuracyChart metrics={metrics} />
          </div>

          {/* Metrics Summary */}
          <MetricsSummaryBar
            currentMetrics={currentMetrics}
            bestEpoch={bestEpoch}
            currentEpoch={currentEpoch}
            totalEpochs={totalEpochs}
          />
        </>
      )}

      {/* No Data Message */}
      {metrics.length === 0 && status === "running" && (
        <div className="p-8 text-center text-gray-500">
          <p>Waiting for training to start...</p>
        </div>
      )}
    </div>
  );
}
