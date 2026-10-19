/**
 * Inference Service for live model predictions and feature schema inspection.
 * @module
 */

import axios from "@/shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "@/constants/Urls";
import type { ModelFeatureSchemaResponse, InferenceResponse } from "@/types/inference";

/**
 * Fetch feature schema, bounds, and test sample records for interactive inference.
 */
export async function getFeatureSchema(jobId: string): Promise<ModelFeatureSchemaResponse> {
  const response = await axios.get<ModelFeatureSchemaResponse>(
    `${BACKEND_MODEL_ANALYSIS}/${jobId}/feature-schema`,
  );
  return response.data;
}

/**
 * Execute real-time inference on a custom feature dictionary.
 */
export async function runInference(
  jobId: string,
  inputs: Record<string, number>,
): Promise<InferenceResponse> {
  const response = await axios.post<InferenceResponse>(
    `${BACKEND_MODEL_ANALYSIS}/${jobId}/predict`,
    { inputs },
  );
  return response.data;
}
