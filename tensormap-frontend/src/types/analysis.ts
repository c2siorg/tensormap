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

/** Single point on a probability calibration / reliability curve. */
export interface CalibrationPoint {
  bin_index: number;
  bin_label: string;
  mean_predicted: number;
  fraction_positives: number | null;
  count: number;
  error: number | null;
}

/** Confidence distribution histogram bin. */
export interface CalibrationHistogramBin {
  bin_index: number;
  bin_start: number;
  bin_end: number;
  bin_label: string;
  count: number;
  frequency: number;
}

/** Per-class calibration analysis data. */
export interface ClassCalibrationData {
  class_index: number;
  class_name: string;
  insufficient_data: boolean;
  brier_score: number | null;
  ece: number | null;
  mce: number | null;
  points: CalibrationPoint[];
  histogram: CalibrationHistogramBin[];
}

/** Response from GET /analysis/{job_id}/calibration-curve. */
export interface CalibrationCurveData {
  classes: ClassCalibrationData[];
  macro_brier_score: number | null;
  macro_ece: number | null;
  macro_mce: number | null;
  is_binary: boolean;
  class_names: string[];
  n_samples: number;
  n_bins: number;
  analysis_type: "classification";
  cached?: boolean;
}
