/**
 * TypeScript types for the Interactive Model Inference Playground.
 * @module
 */

export interface FeatureSchemaItem {
  name: string;
  dtype: "float" | "int" | "categorical" | string;
  min: number | null;
  max: number | null;
  mean: number | null;
  median: number | null;
  step: number | null;
  default_value: number;
}

export interface SampleTestRecord {
  index: number;
  features: Record<string, number>;
  actual_class?: number | null;
  actual_class_name?: string | null;
  actual_value?: number | null;
}

export interface ModelFeatureSchemaResponse {
  job_id: string;
  target_field: string;
  problem_type: "classification" | "regression" | string;
  features: FeatureSchemaItem[];
  class_names?: string[] | null;
  sample_records: SampleTestRecord[];
}

export interface ClassProbability {
  class_index: number;
  class_name: string;
  probability: number;
}

export interface InferenceResponse {
  success: boolean;
  problem_type: "classification" | "regression" | string;
  predicted_class?: number | null;
  predicted_label?: string | null;
  confidence?: number | null;
  predicted_value?: number | null;
  probabilities?: ClassProbability[] | null;
  latency_ms: number;
  input_features: Record<string, number>;
}
