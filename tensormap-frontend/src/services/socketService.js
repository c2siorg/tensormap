/**
 * Socket.IO client for per-job training progress.
 *
 * Training metrics are isolated per job via Socket.IO rooms: a client subscribes
 * to a `job_id` and only receives that job's events (plus a one-shot catch-up of
 * the persisted history). A single shared socket is reused across the app.
 * @module
 */

import { io } from "socket.io-client";
import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import * as strings from "../constants/Strings";

const SUPPORTED_TRAINING_EVENT_VERSION = 1;
const TRAINING_STATUSES = new Set(["pending", "running", "completed", "failed", "cancelled"]);

let socket = null;

/** Lazily create (once) and return the shared training socket. */
export function getTrainingSocket() {
  if (!socket) {
    socket = io(urls.WS_DL_RESULTS, {
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 10000,
      autoConnect: false,
    });
  }
  return socket;
}

/**
 * Subscribe to a job's room and stream its events to `onEvent`.
 *
 * `onEvent` receives validated versioned training envelopes, explicitly shaped
 * legacy training events, or the legacy `{ message, test }` model progress event.
 *
 * @returns {() => void} cleanup that detaches the listener.
 */
export function subscribeToJob(jobId, onEvent) {
  const s = getTrainingSocket();

  const handler = (data) => {
    if (data === null || typeof data !== "object" || Array.isArray(data)) return;

    const hasEnvelope = data.version !== undefined;
    if (hasEnvelope) {
      if (
        data.version !== SUPPORTED_TRAINING_EVENT_VERSION ||
        data.job_id !== jobId ||
        typeof data.timestamp !== "string"
      )
        return;

      if (isValidTrainingEvent(data, true)) onEvent(data);
      return;
    }

    // Accept the old, explicitly shaped job events while older emitters remain
    // in use. If they carry a job id, it must still match this subscription.
    if (data.job_id !== undefined && data.job_id !== jobId) return;
    if (isValidTrainingEvent(data, false)) {
      onEvent(data);
      return;
    }

    // Legacy CustomProgressBar and _model_result emit exactly {message, test}.
    // Keep that documented producer shape for non-job-scoped model progress.
    if (typeof data.message === "string" && Number.isFinite(data.test)) {
      onEvent(data);
    }
  };

  s.on(strings.DL_RESULT_LISTENER, handler);
  // Re-join the room on every (re)connect: a reconnect gives a fresh server-side
  // session, so the previous room membership is gone. Re-subscribing triggers a
  // catch-up so no metrics are missed across the gap.
  const emitSubscribe = () => s.emit("subscribe_job", { job_id: jobId });
  s.on("connect", emitSubscribe);
  if (s.connected) {
    emitSubscribe();
  } else {
    s.connect();
  }

  return () => {
    s.off(strings.DL_RESULT_LISTENER, handler);
    s.off("connect", emitSubscribe);
  };
}

function isFiniteMetric(value) {
  return (
    value === undefined || value === null || (typeof value === "number" && Number.isFinite(value))
  );
}

function isCounter(value, nullable = false) {
  return (nullable && value === null) || (Number.isInteger(value) && value >= 0);
}

function isValidMetricRecord(metric) {
  return (
    metric !== null &&
    typeof metric === "object" &&
    !Array.isArray(metric) &&
    Number.isInteger(metric.epoch) &&
    ["loss", "accuracy", "val_loss", "val_accuracy"].every((key) => isFiniteMetric(metric[key]))
  );
}

function isValidTrainingEvent(data, structured) {
  if (data.type === "metrics") {
    return (
      isValidMetricRecord(data) &&
      (data.phase === undefined ||
        data.phase === "train" ||
        (!structured && data.phase === "validation")) &&
      ["batch", "steps", "validation_batch", "validation_steps"].every(
        (key) => data[key] === undefined || (Number.isInteger(data[key]) && data[key] >= 0),
      ) &&
      (!structured ||
        (data.phase === "train" &&
          ["loss", "accuracy", "val_loss", "val_accuracy"].every((key) => key in data) &&
          ["batch", "steps", "validation_batch", "validation_steps"].every((key) =>
            isCounter(data[key]),
          )))
    );
  }
  if (data.type === "catchup") {
    return (
      TRAINING_STATUSES.has(data.status) &&
      Array.isArray(data.metrics) &&
      data.metrics.every(isValidMetricRecord) &&
      (data.started_at === undefined ||
        data.started_at === null ||
        typeof data.started_at === "string") &&
      (!structured || "started_at" in data)
    );
  }
  if (data.type === "status") {
    return (
      TRAINING_STATUSES.has(data.status) &&
      (data.error === undefined || typeof data.error === "string") &&
      (!structured ||
        (data.phase === null &&
          data.epoch === null &&
          isCounter(data.batch, true) &&
          isCounter(data.steps, true)))
    );
  }
  return false;
}

/** Leave a job's room so this client stops receiving its events. */
export function unsubscribeFromJob(jobId) {
  if (socket && jobId) {
    socket.emit("unsubscribe_job", { job_id: jobId });
  }
}

/** Request cancellation of a training job (DELETE /model/training-job/:id). */
export function cancelJob(jobId) {
  return axios.delete(`${urls.BACKEND_TRAINING_JOB}/${jobId}`);
}
