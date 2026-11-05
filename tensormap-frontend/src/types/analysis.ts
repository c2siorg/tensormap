/**
 * TypeScript types for interpretability analysis responses.
 * @module
 */

/** Shape of a single feature's importance data. */
export interface FeatureImportance {
  feature: string;
  importance: number;
  std: number;
}

/** Response from GET /analysis/{job_id}/feature-importance (200). */
export interface FeatureImportanceData {
  features: string[];
  importances_mean: number[];
  importances_std: number[];
  analysis_type: "feature_importance";
  n_samples_used: number;
  n_repeats: number;
  cached: boolean;
}

/** Response from GET /analysis/{job_id}/confusion-matrix (200). */
export interface ConfusionMatrixData {
  confusion_matrix: number[][];
  classification_report: Record<
    string,
    {
      precision: number;
      recall: number;
      "f1-score": number;
      support: number;
    }
  >;
  class_names: string[];
  overall_accuracy: number;
  n_samples: number;
  analysis_type: "classification";
  cached: boolean;
}

/** Response from GET /analysis/{job_id}/feature-importance (202). */
export interface ComputingStatus {
  status: "computing";
}

/** Response from regression analysis. */
export interface RegressionAnalysisData {
  residuals: number[];
  y_pred: number[];
  y_true: number[];
  mae: number;
  mse: number;
  analysis_type: "regression";
  cached: boolean;
}

/** Single prediction record. */
export interface Prediction {
  index: number;
  actual_class: number;
  actual_class_name: string;
  predicted_class: number;
  predicted_class_name: string;
  confidence: number;
  probabilities: number[];
  features: Record<string, number>;
  is_correct: boolean;
}

/** Response from GET /analysis/{job_id}/predictions. */
export interface PredictionsData {
  total: number;
  offset: number;
  limit: number;
  predictions: Prediction[];
}

/** Layer weight histogram bin for LayerDiagnostics. */
export interface LayerWeightHistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

/** Detailed layer weight metrics and health status. */
export interface LayerDiagnostics {
  layer_name: string;
  layer_type: string;
  shape: number[];
  param_count: number;
  l2_norm: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  dead_weights_ratio: number;
  status: "HEALTHY" | "VANISHING_WEIGHTS" | "EXPLODING_WEIGHTS" | "HIGH_SPARSITY" | "NAN_OR_INF";
  histogram: LayerWeightHistogramBin[];
  bias_l2_norm?: number;
  bias_mean?: number;
}

/** Response from GET /analysis/{job_id}/layer-diagnostics. */
export interface LayerDiagnosticsData {
  job_id: string;
  overall_status: "HEALTHY" | "WARNING" | "CRITICAL";
  total_layers: number;
  healthy_layers: number;
  total_parameters: number;
  layers: LayerDiagnostics[];
}

/** Fairness metrics for a single demographic subgroup. */
export interface GroupFairnessMetrics {
  group: string;
  sample_count: number;
  selection_rate: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  true_positive_rate: number;
  false_positive_rate: number;
}

/** Disparity measurements comparing a subgroup against privileged group. */
export interface GroupDisparity {
  group: string;
  compared_to: string;
  disparate_impact_ratio: number;
  demographic_parity_difference: number;
  equal_opportunity_difference: number;
  average_odds_difference: number;
  passes_80_percent_rule: boolean;
}

/** Response from GET /analysis/{job_id}/fairness. */
export interface FairnessAuditData {
  job_id: string;
  sensitive_feature: string;
  privileged_group: string;
  group_metrics: Record<string, GroupFairnessMetrics>;
  disparities: Record<string, GroupDisparity>;
  overall_summary: {
    compliant_with_80_percent_rule: boolean;
    groups_evaluated: number;
    total_samples: number;
  };
}
