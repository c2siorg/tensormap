/**
 * Tuning service — start, monitor, cancel, and apply hyperparameter tuning.
 *
 * Provides both HTTP API calls and Socket.IO subscription for real-time
 * tuning progress.  Uses the shared training socket instance.
 * @module
 */

import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import { getTrainingSocket } from "./socketService";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TuningSearchSpace {
  [key: string]:
    | (string | number)[]
    | { type: "log_uniform" | "uniform"; min: number; max: number };
}

export interface TuningStartConfig {
  strategy?: "grid" | "random";
  search_space: TuningSearchSpace;
  max_trials?: number;
  metric?: string;
  direction?: "maximize" | "minimize";
  early_stop_threshold?: number | null;
}

export interface TuningStartResponse {
  tuning_id: string;
  status: string;
  estimated_seconds: number;
  n_trials: number;
}

export interface TuningTrialSummary {
  job_id: string;
  status: string;
  hyperparams: Record<string, unknown> | null;
  metric_value: number | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface TuningSessionDetail {
  tuning_id: string;
  model_id: number;
  strategy: string;
  search_space: TuningSearchSpace;
  max_trials: number;
  metric: string;
  direction: string;
  early_stop_threshold: number | null;
  best_job_id: string | null;
  best_hyperparams: Record<string, unknown> | null;
  status: string;
  total_trials: number;
  completed_trials: number;
  early_stopped: boolean;
  created_at: string | null;
  completed_at: string | null;
  trials: TuningTrialSummary[];
}

export interface TuningProgress {
  type: "tuning_progress" | "tuning_complete" | "tuning_catchup";
  trial?: number;
  total?: number;
  job_id?: string;
  hyperparams?: Record<string, unknown>;
  metric?: number | null;
  best_metric?: number | null;
  best_job_id?: string | null;
  status?: string;
  error?: string | null;
  // catch-up fields
  total_trials?: number;
  completed_trials?: number;
  early_stopped?: boolean;
  trials?: TuningTrialSummary[];
}

// ---------------------------------------------------------------------------
// HTTP API
// ---------------------------------------------------------------------------

/** Start a tuning session for a model (POST, returns 202). */
export function startTuning(modelName: string, config: TuningStartConfig) {
  return axios.post<{ success: boolean; data: TuningStartResponse }>(
    `${urls.BACKEND_MODEL_TUNING}/${encodeURIComponent(modelName)}`,
    {
      model_name: modelName,
      ...config,
    },
  );
}

/** Get the full status of a tuning session. */
export function getTuningSession(tuningId: string) {
  return axios.get<{ success: boolean; data: TuningSessionDetail }>(
    `${urls.BACKEND_MODEL_TUNING}/${tuningId}`,
  );
}

/** Cancel a running tuning session (DELETE, returns 204). */
export function cancelTuning(tuningId: string) {
  return axios.delete(`${urls.BACKEND_MODEL_TUNING}/${tuningId}`);
}

/** Apply the best trial's hyperparams to the model config (POST). */
export function applyBestParams(tuningId: string) {
  return axios.post(`${urls.BACKEND_MODEL_TUNING}/${tuningId}/apply-best`);
}

// ---------------------------------------------------------------------------
// Socket.IO subscription
// ---------------------------------------------------------------------------

/**
 * Subscribe to a tuning session's progress room.
 *
 * `onProgress` receives structured payloads:
 *   - `{ type: "tuning_catchup", ... }` — replay of current state
 *   - `{ type: "tuning_progress", trial, total, hyperparams, metric, ... }`
 *   - `{ type: "tuning_complete", status, best_job_id, best_metric }`
 *
 * @returns cleanup function that detaches the listener.
 */
export function subscribeToTuning(
  tuningId: string,
  onProgress: (data: TuningProgress) => void,
): () => void {
  const s = getTrainingSocket();

  const handler = (data: TuningProgress) => {
    if (
      data &&
      (data.type === "tuning_progress" ||
        data.type === "tuning_complete" ||
        data.type === "tuning_catchup")
    ) {
      onProgress(data);
    }
  };
  s.on("tuning_progress", handler);

  // Re-subscribe on reconnect (server forgets room membership).
  const emitSubscribe = () =>
    s.emit("subscribe_tuning", { tuning_id: tuningId });
  s.on("connect", emitSubscribe);
  if (s.connected) {
    emitSubscribe();
  } else {
    s.connect();
  }

  return () => {
    s.off("tuning_progress", handler);
    s.off("connect", emitSubscribe);
  };
}

/** Leave a tuning session's room so this client stops receiving its events. */
export function unsubscribeFromTuning(tuningId: string): void {
  const s = getTrainingSocket();
  if (s && tuningId) {
    s.emit("unsubscribe_tuning", { tuning_id: tuningId });
  }
}
