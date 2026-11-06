export type RelevanceTier = "CRITICAL" | "INFORMATIVE" | "WEAK" | "CONSTANT_OR_REDUNDANT";

export interface FeatureMetricItem {
  feature: string;
  mutual_info: number;
  f_statistic?: number | null;
  p_value?: number | null;
  variance: number;
  normalized_score: number;
  tier: RelevanceTier;
  is_constant: boolean;
  suggested_action: "keep" | "review" | "prune";
}

export interface FeatureSelectionAnalyzeRequest {
  target_field?: string;
  problem_type?: "classification" | "regression";
  n_neighbors?: number;
}

export interface FeatureSelectionAnalyzeResponse {
  file_id: string;
  target_field: string;
  problem_type: string;
  total_samples: number;
  total_features: number;
  critical_features_count: number;
  informative_features_count: number;
  weak_features_count: number;
  constant_features_count: number;
  features: FeatureMetricItem[];
}

export interface ApplyFeatureSelectionRequest {
  selected_features?: string[];
  top_k?: number;
  min_score?: number;
}

export interface ApplyFeatureSelectionResponse {
  file_id: string;
  target_field: string;
  retained_features: string[];
  dropped_features: string[];
  total_columns_remaining: number;
  backup_created: boolean;
  message: string;
}
