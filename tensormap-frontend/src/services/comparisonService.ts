/**
 * Comparison service — fetch and compare training run metrics.
 * @module
 */

import axios from "../shared/Axios";
import * as urls from "../constants/Urls";

export interface JobMetric {
  epoch: number;
  loss?: number;
  val_loss?: number;
  accuracy?: number;
  val_accuracy?: number;
  [key: string]: number | undefined;
}

export interface ComparisonJob {
  job_id: string;
  hyperparams: {
    optimizer?: string;
    lr?: number;
    batch_size?: number;
    epochs?: number;
    [key: string]: any;
  };
  status: string;
  started_at: string | null;
  completed_at: string | null;
  metrics: JobMetric[];
}

export interface ComparisonResponse {
  jobs: ComparisonJob[];
  metric: string;
}

/**
 * Compare multiple training jobs.
 * @param jobIds - Array of job IDs to compare (max 5)
 * @param metric - Metric to compare (default: val_loss)
 */
export function compareJobs(jobIds: string[], metric: string = "val_loss") {
  const jobIdsStr = jobIds.join(",");
  return axios.get<{ success: boolean; data: ComparisonResponse }>(
    `/model/compare?job_ids=${jobIdsStr}&metric=${metric}`,
  );
}

/**
 * Get list of training jobs for a model.
 * @param modelName - Name of the model
 */
export function getTrainingJobs(modelName: string) {
  return axios.get<{
    success: boolean;
    data: Array<{
      job_id: string;
      model_id: number;
      status: string;
      started_at: string | null;
      completed_at: string | null;
    }>;
  }>(`/model/training-jobs?model_name=${modelName}`);
}
