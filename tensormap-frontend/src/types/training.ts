/**
 * TypeScript types for training metrics and state management.
 * @module
 */

export interface EpochMetric {
  epoch: number;
  loss: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;

  phase?: "train" | "validation";
  batch?: number;
  steps?: number;
  timestamp?: string;
  job_id?: string;
  version?: number;
}

export interface TrainingState {
  status: "idle" | "pending" | "running" | "completed" | "failed" | "cancelled";
  jobId: string | null;
  metrics: EpochMetric[];
  totalEpochs: number;
  currentEpoch: number;
  startedAt: string | null;
  error: string | null;
}

export interface TrainingMetricsChartProps {
  jobId: string;
  totalEpochs: number;
  onComplete?: (jobId: string) => void;
}

export interface TrainingHeaderProps {
  jobId: string;
  status: TrainingState["status"];
  currentEpoch: number;
  totalEpochs: number;
  elapsedTime: number;
  eta: number;
  onStop: () => void;
  cancelRequested: boolean;
}

export interface ChartProps {
  metrics: EpochMetric[];
}

export interface MetricsSummaryBarProps {
  currentMetrics: EpochMetric | null;
  bestEpoch: { epoch: number; val_loss: number } | null;
  currentEpoch: number;
  totalEpochs: number;
}
