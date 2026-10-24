/**
 * Hook for managing training metrics state with Socket.IO and fallback polling.
 *
 * Features:
 * - Subscribes to Socket.IO job room on mount
 * - Appends new EpochMetric on each "metrics" event
 * - Populates from catchup event on subscribe
 * - Fetches full history from GET /api/model/training-job/{id}/metrics on page reload
 * - Fallback polling when Socket.IO disconnects
 * - Returns: { metrics, status, currentEpoch, totalEpochs, cancel, cancelRequested, startedAt, error }
 * @module
 */

import { useState, useEffect, useRef, useCallback } from "react";
import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import {
  subscribeToJob,
  unsubscribeFromJob,
  cancelJob,
  getTrainingSocket,
} from "../services/socketService";
import {
  EpochMetric,
  EpochMetricUpdate,
  LegacyTrainingMetricsEvent,
  TrainingMetricsEvent,
  TrainingSocketEvent,
  TrainingState,
} from "../types/training";
import logger from "../shared/logger";

interface UseTrainingMetricsReturn {
  metrics: EpochMetric[];
  status: TrainingState["status"];
  currentEpoch: number;
  totalEpochs: number;
  cancel: () => void;
  cancelRequested: boolean;
  startedAt: string | null;
  error: string | null;
  isCompleted: boolean;
  isCancelled: boolean;
}

export function mergeEpochMetric(
  metrics: EpochMetric[],
  incoming: EpochMetricUpdate,
): EpochMetric[] {
  const existingIndex = metrics.findIndex((metric) => metric.epoch === incoming.epoch);
  if (existingIndex === -1) {
    return [...metrics, normalizeEpochMetric(incoming)];
  }

  const merged = { ...metrics[existingIndex] };
  mergeDefinedValues(merged, incoming);
  return metrics.map((metric, index) => (index === existingIndex ? merged : metric));
}

function normalizeEpochMetric(incoming: EpochMetricUpdate): EpochMetric {
  const normalized: EpochMetric = { epoch: incoming.epoch };
  mergeDefinedValues(normalized, incoming);
  return normalized;
}

export function mergeEpochMetrics(current: EpochMetric[], incoming: EpochMetric[]): EpochMetric[] {
  const merged = incoming.reduce(
    (result, metric) => mergeEpochMetric(result, metric),
    [] as EpochMetric[],
  );
  // Existing state may include live events received while catch-up or polling
  // was in flight, so it takes precedence for fields already present.
  return current.reduce((result, metric) => mergeEpochMetric(result, metric), merged);
}

export function shouldPollTrainingMetrics(
  jobId: string | null,
  isSocketConnected: boolean,
  status: TrainingState["status"],
): boolean {
  return Boolean(jobId) && !isSocketConnected && status === "running";
}

export function toEpochMetricUpdate(
  event: TrainingMetricsEvent | LegacyTrainingMetricsEvent,
): EpochMetricUpdate {
  const isLegacyValidation = event.phase === "validation";
  return {
    epoch: event.epoch,
    loss: isLegacyValidation ? null : event.loss,
    accuracy: isLegacyValidation ? null : event.accuracy,
    val_loss: isLegacyValidation ? (event.val_loss ?? event.loss) : event.val_loss,
    val_accuracy: isLegacyValidation ? (event.val_accuracy ?? event.accuracy) : event.val_accuracy,
    phase: event.phase === "train" ? "train" : undefined,
    batch: isLegacyValidation ? undefined : event.batch,
    steps: isLegacyValidation ? undefined : event.steps,
    validation_batch: isLegacyValidation
      ? (event.validation_batch ?? event.batch)
      : event.validation_batch,
    validation_steps: isLegacyValidation
      ? (event.validation_steps ?? event.steps)
      : event.validation_steps,
    timestamp: event.timestamp,
    job_id: event.job_id,
    version: event.version,
  };
}

function mergeDefinedValues(target: EpochMetric, incoming: EpochMetricUpdate): void {
  if (incoming.loss != null) target.loss = incoming.loss;
  if (incoming.accuracy != null) target.accuracy = incoming.accuracy;
  if (incoming.val_loss != null) target.val_loss = incoming.val_loss;
  if (incoming.val_accuracy != null) target.val_accuracy = incoming.val_accuracy;
  if (incoming.phase !== undefined) target.phase = incoming.phase;
  if (incoming.batch != null) target.batch = incoming.batch;
  if (incoming.steps != null) target.steps = incoming.steps;
  if (incoming.validation_batch != null) target.validation_batch = incoming.validation_batch;
  if (incoming.validation_steps != null) target.validation_steps = incoming.validation_steps;
  if (incoming.timestamp !== undefined) target.timestamp = incoming.timestamp;
  if (incoming.job_id !== undefined) target.job_id = incoming.job_id;
  if (incoming.version != null) target.version = incoming.version;
}

export function useTrainingMetrics(
  jobId: string | null,
  totalEpochs: number,
): UseTrainingMetricsReturn {
  const [metrics, setMetrics] = useState<EpochMetric[]>([]);
  const [status, setStatus] = useState<TrainingState["status"]>("idle");
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSocketConnected, setIsSocketConnected] = useState(true);

  const cleanupRef = useRef<(() => void) | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch metrics from API (fallback or initial load)
  const fetchMetrics = useCallback(async () => {
    if (!jobId) return;

    try {
      const response = await axios.get(`${urls.BACKEND_TRAINING_JOB}/${jobId}/metrics`);
      // Backend returns {success, message, data: [...metrics...]}
      if (response.data && response.data.data && Array.isArray(response.data.data)) {
        setMetrics((previous) => mergeEpochMetrics(previous, response.data.data));
        setCurrentEpoch((previous) => Math.max(previous, response.data.data.length));
      }
    } catch (err) {
      logger.error("Failed to fetch metrics:", err);
    }
  }, [jobId]);

  // Handle Socket.IO events
  const handleJobEvent = useCallback((data: TrainingSocketEvent) => {
    if ("type" in data && data.type === "catchup") {
      // Merge catch-up with live events that may have arrived after room join.
      setMetrics((previous) => mergeEpochMetrics(previous, data.metrics));
      setCurrentEpoch((previous) => Math.max(previous, data.metrics.length));
      setStatus(data.status || "running");
      if (data.started_at) {
        setStartedAt(data.started_at);
      }
    } else if ("type" in data && data.type === "metrics") {
      // Live metrics use the same one-record-per-epoch shape as catch-up.
      const newMetric = toEpochMetricUpdate(data);
      setMetrics((prev) => mergeEpochMetric(prev, newMetric));
      setCurrentEpoch(data.epoch);
      setStatus("running");
    } else if ("type" in data && data.type === "status") {
      // Terminal status
      setStatus(data.status);
      if (data.status === "cancelled") {
        setCancelRequested(false);
      }
      if (data.error) {
        setError(data.error);
      }
    }
  }, []);

  // Setup Socket.IO subscription
  useEffect(() => {
    if (!jobId) return;

    const socket = getTrainingSocket();

    // Socket connection state handlers
    const handleConnect = () => {
      setIsSocketConnected(true);
      // Stop polling when socket reconnects
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };

    const handleDisconnect = () => {
      setIsSocketConnected(false);
    };

    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);

    // Subscribe to job room
    const cleanup = subscribeToJob(jobId, handleJobEvent);
    cleanupRef.current = cleanup;

    // Initial fetch in case of page reload
    fetchMetrics();

    return () => {
      cleanup();
      unsubscribeFromJob(jobId);
      socket.off("connect", handleConnect);
      socket.off("disconnect", handleDisconnect);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [jobId, handleJobEvent, fetchMetrics]);

  // Fallback polling when Socket.IO disconnects
  useEffect(() => {
    if (!shouldPollTrainingMetrics(jobId, isSocketConnected, status)) {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    // Start polling if socket is disconnected and training is running
    if (!isSocketConnected && status === "running") {
      pollingIntervalRef.current = setInterval(() => {
        fetchMetrics();
      }, 3000);
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [jobId, isSocketConnected, status, fetchMetrics]);

  // Cancel training
  const cancel = useCallback(() => {
    if (!jobId || cancelRequested) return;

    setCancelRequested(true);
    cancelJob(jobId).catch((err) => {
      logger.error("Failed to cancel training:", err);
      setCancelRequested(false);
    });
  }, [jobId, cancelRequested]);

  return {
    metrics,
    status,
    currentEpoch,
    totalEpochs,
    cancel,
    cancelRequested,
    startedAt,
    error,
    isCompleted: status === "completed",
    isCancelled: status === "cancelled",
  };
}
