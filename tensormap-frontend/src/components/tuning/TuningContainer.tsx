/**
 * Tuning Container Component
 * Main orchestrator for the hyperparameter tuning workflow.
 * @module
 */

import { useState, useCallback } from "react";
import TuningConfigForm from "./TuningConfigForm";
import TuningProgressPanel from "./TuningProgressPanel";
import ApplyBestBanner from "./ApplyBestBanner";
import {
  startTuning,
  cancelTuning,
  applyBestParams,
  getTuningSession,
  TuningSearchSpace,
  TuningSessionDetail,
} from "@/services/tuningService";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface TuningContainerProps {
  modelName: string;
}

export default function TuningContainer({ modelName }: TuningContainerProps) {
  const [tuningId, setTuningId] = useState<string | null>(null);
  const [totalTrials, setTotalTrials] = useState(0);
  const [metric, setMetric] = useState("val_accuracy");
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [sessionDetail, setSessionDetail] = useState<TuningSessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCompleted, setIsCompleted] = useState(false);

  const handleStart = useCallback(
    async (config: {
      strategy: "grid" | "random";
      search_space: TuningSearchSpace;
      max_trials: number;
      metric: string;
      direction: "maximize" | "minimize";
      early_stop_threshold: number | null;
    }) => {
      setError(null);
      try {
        const response = await startTuning(modelName, config);
        const data = response.data.data;

        setTuningId(data.tuning_id);
        setTotalTrials(data.n_trials);
        setMetric(config.metric);
        setDirection(config.direction);
        setIsCompleted(false);
      } catch (err: any) {
        setError(
          err.response?.data?.message || err.message || "Failed to start tuning session",
        );
      }
    },
    [modelName],
  );

  const handleCancel = useCallback(async () => {
    if (!tuningId) return;
    try {
      await cancelTuning(tuningId);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to cancel tuning");
    }
  }, [tuningId]);

  const handleComplete = useCallback(async () => {
    if (!tuningId) return;
    setIsCompleted(true);

    // Fetch full session details to get best params
    try {
      const response = await getTuningSession(tuningId);
      setSessionDetail(response.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch session details");
    }
  }, [tuningId]);

  const handleApplyBest = useCallback(async () => {
    if (!tuningId) return;
    await applyBestParams(tuningId);
  }, [tuningId]);

  const estimateTime = useCallback(
    (space: TuningSearchSpace, strategy: string, maxTrials: number) => {
      // Simple estimate: 2 minutes per trial
      return maxTrials * 120;
    },
    [],
  );

  return (
    <div className="space-y-6">
      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Configuration Form (only show if no active tuning) */}
      {!tuningId && (
        <TuningConfigForm
          modelName={modelName}
          onStart={handleStart}
          estimateTime={estimateTime}
        />
      )}

      {/* Progress Panel (show during tuning) */}
      {tuningId && !isCompleted && (
        <TuningProgressPanel
          tuningId={tuningId}
          totalTrials={totalTrials}
          metric={metric}
          direction={direction}
          onCancel={handleCancel}
          onComplete={handleComplete}
        />
      )}

      {/* Apply Best Banner (show after completion) */}
      {isCompleted && sessionDetail && sessionDetail.best_hyperparams && (
        <ApplyBestBanner
          bestParams={sessionDetail.best_hyperparams}
          metric={metric}
          metricValue={
            sessionDetail.trials.find((t) => t.job_id === sessionDetail.best_job_id)
              ?.metric_value || 0
          }
          onApply={handleApplyBest}
        />
      )}

      {/* Show progress panel even after completion for reference */}
      {isCompleted && tuningId && (
        <TuningProgressPanel
          tuningId={tuningId}
          totalTrials={totalTrials}
          metric={metric}
          direction={direction}
          onCancel={handleCancel}
        />
      )}
    </div>
  );
}
