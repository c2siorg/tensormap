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
 * `onEvent` receives structured payloads of the form:
 *   - `{ type: "catchup", status, metrics: [...] }` — replay of persisted history
 *   - `{ type: "metrics", epoch, loss, accuracy, val_loss, val_accuracy }`
 *   - `{ type: "status", status, error? }` — terminal state (completed/failed/cancelled)
 *
 * @returns {() => void} cleanup that detaches the listener.
 */
export function subscribeToJob(jobId, onEvent) {
  const s = getTrainingSocket();

  const handler = (data) => {
    if (data && (data.type === "metrics" || data.type === "catchup" || data.type === "status")) {
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
