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
import { EpochMetric, TrainingState } from "../types/training";
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
        setMetrics(response.data.data);
        setCurrentEpoch(response.data.data.length);
      }
    } catch (err) {
      logger.error("Failed to fetch metrics:", err);
    }
  }, [jobId]);

  // Handle Socket.IO events
  const handleJobEvent = useCallback((data: any) => {
    if (data.type === "catchup") {
      // Populate from catchup event
      setMetrics(data.metrics || []);
      setCurrentEpoch((data.metrics || []).length);
      setStatus(data.status || "running");
      if (data.started_at) {
        setStartedAt(data.started_at);
      }
    } else if (data.type === "metrics") {
      // Append new metric
     const newMetric: EpochMetric = {
      epoch: data.epoch,
      loss: data.loss,
      accuracy: data.accuracy,
      val_loss: data.val_loss,
      val_accuracy: data.val_accuracy,
      phase: data.phase,
      batch: data.batch,
      steps: data.steps,
      timestamp: data.timestamp,
      job_id: data.job_id,
      version: data.version,
    };
      setMetrics((prev) => {
        const existing = prev.find((metric) => metric.epoch === data.epoch);

        if (!existing) {
          return [...prev, newMetric];
        }

        return prev.map((metric) =>
          metric.epoch === data.epoch
            ? { ...metric, ...newMetric }
            : metric
        );
      });
      setCurrentEpoch(data.epoch);
      setStatus("running");
    } else if (data.type === "status") {
      // Terminal status
      setStatus(data.status);
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
    if (!jobId || !isSocketConnected || status !== "running") {
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
  };
}
