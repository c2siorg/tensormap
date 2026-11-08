/**
 * TypeScript types for training run reproducibility metadata.
 * Corresponds to backend schemas in app/schemas/training.py
 * @module
 */

export interface DatasetProvenance {
  file_id: string;
  filename: string | null;
  dataset_type: "csv" | "zip";
  fingerprint_algorithm: "sha256";
  fingerprint: string;
  fingerprint_scope: "file_bytes" | "image_tree_manifest_v1";
  size_bytes: number;
}

export interface TrainingConfigurationSnapshot {
  problem_type_id: number | null;
  problem_type: string | null;
  target_column: string | null;
  feature_columns: string[] | null;
  class_names: string[] | null;
  model_architecture: Record<string, any> | null;
  architecture_sha256: string | null;
  optimizer: any;
  loss: any;
  metrics: string[];
  epochs: number | null;
  batch_size: number | null;
  training_split_percent: number | null;
  holdout_split_percent: number | null;
  requested_training_split_percent: number | null;
  requested_holdout_split_percent: number | null;
  actual_training_rows: number | null;
  actual_holdout_rows: number | null;
  actual_training_split_percent: number | null;
  actual_holdout_split_percent: number | null;
  holdout_role: "validation" | "evaluation" | null;
  configured_hyperparameters: Record<string, any>;
  effective_hyperparameters: Record<string, any>;
}

export interface RandomnessSnapshot {
  configured_seed: number | null;
  effective_seeds: Record<string, number | null>;
  applied_seeds: string[];
  global_framework_seed_applied: boolean;
  deterministic_execution_guaranteed: boolean;
}

export interface RuntimeVersions {
  python: string;
  tensorflow: string | null;
  keras: string | null;
  numpy: string | null;
  pandas: string | null;
}

export interface ReproducibilitySnapshot {
  schema_version: 1 | 2;
  job_id: string;
  captured_at: string;
  dataset: DatasetProvenance;
  training_configuration: TrainingConfigurationSnapshot;
  randomness: RandomnessSnapshot;
  runtime_versions: RuntimeVersions;
}

export interface TrainingJobDetails {
  job_id: string;
  model_id: number;
  status: string;
  hyperparams: Record<string, any> | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  latest_metrics: Record<string, any> | null;
  reproducibility_metadata: ReproducibilitySnapshot | null;
}
