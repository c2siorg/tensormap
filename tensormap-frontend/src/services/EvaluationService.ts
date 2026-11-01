/**
 * Evaluation service for holdout test dataset benchmarking and batch predictions.
 * @module
 */

import axios from "../shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "../constants/Urls";
import { BatchEvaluationResponse, BatchPredictionResponse } from "../types/evaluation";

/**
 * Runs evaluation on a holdout test dataset.
 */
export async function evaluateHoldoutDataset(
  jobId: string,
  fileId: string,
  targetField?: string,
  batchSize: number = 256,
): Promise<BatchEvaluationResponse> {
  const payload: Record<string, any> = {
    file_id: fileId,
    batch_size: batchSize,
  };
  if (targetField) {
    payload.target_field = targetField;
  }

  const response = await axios.post<BatchEvaluationResponse>(
    `${BACKEND_MODEL_ANALYSIS}/${jobId}/evaluate`,
    payload,
  );
  return response.data;
}

/**
 * Runs batch prediction on an uploaded dataset and returns preview rows.
 */
export async function runBatchPrediction(
  jobId: string,
  fileId: string,
  batchSize: number = 256,
): Promise<BatchPredictionResponse> {
  const response = await axios.post<BatchPredictionResponse>(
    `${BACKEND_MODEL_ANALYSIS}/${jobId}/batch-predict`,
    {
      file_id: fileId,
      batch_size: batchSize,
    },
  );
  return response.data;
}

/**
 * Downloads batch predictions as a CSV file.
 */
export async function downloadBatchPredictionsCsv(
  jobId: string,
  fileId: string,
  batchSize: number = 256,
): Promise<void> {
  const response = await axios.post(
    `${BACKEND_MODEL_ANALYSIS}/${jobId}/batch-predict/export`,
    {
      file_id: fileId,
      batch_size: batchSize,
    },
    {
      responseType: "blob",
    },
  );

  const blob = new Blob([response.data], { type: "text/csv" });
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.setAttribute("download", `predictions_${jobId.slice(0, 8)}.csv`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(downloadUrl);
}
