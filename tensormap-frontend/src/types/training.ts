/**
 * TypeScript types for training metrics and state management.
 * @module
 */

export interface EpochMetric {
  epoch: number;
  loss?: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;

  phase?: "train";
  batch?: number;
  steps?: number;
  validation_batch?: number;
  validation_steps?: number;
  timestamp?: string;
  job_id?: string;
  version?: number;
}

/** Partial metric values received from a live event before normalization. */
export type EpochMetricUpdate = Omit<
  Partial<EpochMetric>,
  "epoch" | "loss" | "accuracy" | "val_loss" | "val_accuracy"
> & {
  epoch: number;
  loss?: number | null;
  accuracy?: number | null;
  val_loss?: number | null;
  val_accuracy?: number | null;
};

export interface TrainingState {
  status: "idle" | "pending" | "running" | "completed" | "failed" | "cancelled";
  jobId: string | null;
  metrics: EpochMetric[];
  totalEpochs: number;
  currentEpoch: number;
  startedAt: string | null;
  error: string | null;
}

export type TrainingJobStatus = Exclude<TrainingState["status"], "idle">;

export interface TrainingMetricsEvent {
  version: 1;
  type: "metrics";
  job_id: string;
  timestamp: string;
  phase: "train";
  epoch: number;
  loss: number | null;
  accuracy: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  batch: number;
  steps: number;
  validation_batch: number;
  validation_steps: number;
}

export interface TrainingCatchupEvent {
  version: 1;
  type: "catchup";
  job_id: string;
  timestamp: string;
  status: TrainingJobStatus;
  started_at: string | null;
  metrics: EpochMetric[];
}

export interface TrainingStatusEvent {
  version: 1;
  type: "status";
  job_id: string;
  timestamp: string;
  status: TrainingJobStatus;
  phase: null;
  epoch: null;
  batch: number | null;
  steps: number | null;
  error?: string;
}

/** Previously supported training payloads emitted before envelope version 1. */
export interface LegacyTrainingMetricsEvent {
  type: "metrics";
  epoch: number;
  loss?: number | null;
  accuracy?: number | null;
  val_loss?: number | null;
  val_accuracy?: number | null;
  phase?: "train" | "validation";
  batch?: number;
  steps?: number;
  validation_batch?: number;
  validation_steps?: number;
  job_id?: string;
  timestamp?: string;
  version?: never;
}

export interface LegacyTrainingCatchupEvent {
  type: "catchup";
  status: TrainingJobStatus;
  metrics: EpochMetric[];
  started_at?: string | null;
  job_id?: string;
  timestamp?: string;
  version?: never;
}

export interface LegacyTrainingStatusEvent {
  type: "status";
  status: TrainingJobStatus;
  error?: string;
  job_id?: string;
  timestamp?: string;
  version?: never;
}

export interface LegacyModelProgressEvent {
  message: string;
  test: number;
}

export type TrainingSocketEvent =
  | TrainingMetricsEvent
  | TrainingCatchupEvent
  | TrainingStatusEvent
  | LegacyTrainingMetricsEvent
  | LegacyTrainingCatchupEvent
  | LegacyTrainingStatusEvent
  | LegacyModelProgressEvent;

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
