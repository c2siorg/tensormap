/**
 * TypeScript interface definitions for holdout test dataset evaluation and batch prediction.
 */

export interface TopErrorSample {
  index: number;
  features: Record<string, number | string>;
  actual_label: string;
  predicted_label: string;
  confidence: number | null;
  loss_or_residual: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  f1_weighted: number;
  log_loss?: number | null;
  confusion_matrix?: number[][];
  class_names?: string[];
}

export interface RegressionMetrics {
  mse: number;
  rmse: number;
  mae: number;
  r2?: number | null;
  mape?: number | null;
  max_error: number;
}

export interface BatchEvaluationResponse {
  job_id: string;
  file_id: string;
  problem_type: "classification" | "regression" | "image_classification";
  total_samples: number;
  evaluated_at: string;
  metrics: ClassificationMetrics | RegressionMetrics | Record<string, any>;
  top_errors: TopErrorSample[];
}

export interface BatchPredictionResponse {
  job_id: string;
  file_id: string;
  problem_type: string;
  total_rows: number;
  columns: string[];
  preview_rows: Record<string, any>[];
}

export interface FileItem {
  file_id: string;
  file_name: string;
  file_type: string;
}
