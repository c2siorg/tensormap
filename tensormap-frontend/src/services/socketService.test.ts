/**
 * Test suite for socketService
 * @module
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

const mockListeners = new Map<string, Function>();
const mockSocket = {
  on: vi.fn((event: string, callback: Function) => mockListeners.set(event, callback)),
  off: vi.fn(),
  emit: vi.fn(),
  connect: vi.fn(),
  disconnect: vi.fn(),
  connected: true,
};

// Mock axios - must be before imports
vi.mock("../shared/Axios", () => ({
  default: {
    delete: vi.fn(() => Promise.resolve({ data: { success: true } })),
  },
}));

// Mock socket.io-client - must be before imports
vi.mock("socket.io-client", () => ({
  io: vi.fn(() => mockSocket),
}));

import { subscribeToJob, unsubscribeFromJob, cancelJob, getTrainingSocket } from "./socketService";
import axios from "../shared/Axios";

describe("socketService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockListeners.clear();
  });

  describe("subscribeToJob", () => {
    it("subscribes to job and returns cleanup function", () => {
      const jobId = "test-job-123";
      const onEvent = vi.fn();

      const cleanup = subscribeToJob(jobId, onEvent);

      // Should return a cleanup function
      expect(typeof cleanup).toBe("function");
    });

    it("accepts supported structured event types and the real legacy progress shape", () => {
      const jobId = "test-job-123";
      const onEvent = vi.fn();

      const cleanup = subscribeToJob(jobId, onEvent);

      const handler = mockListeners.get("result :::")!;
      const envelope = { version: 1, job_id: jobId, timestamp: "2026-10-13T12:00:00Z" };
      handler({
        ...envelope,
        type: "metrics",
        phase: "train",
        epoch: 1,
        loss: null,
        accuracy: 0,
        val_loss: null,
        val_accuracy: null,
        batch: 0,
        steps: 0,
        validation_batch: 0,
        validation_steps: 0,
      });
      handler({ ...envelope, type: "catchup", status: "running", started_at: null, metrics: [] });
      handler({
        ...envelope,
        type: "status",
        status: "completed",
        phase: null,
        epoch: null,
        batch: 0,
        steps: 0,
      });
      handler({ type: "metrics", epoch: 2, loss: null, val_loss: 0.25 });
      handler({ type: "metrics", phase: "validation", epoch: 2, loss: 0.25, accuracy: 0.8 });
      handler({ type: "catchup", status: "running", metrics: [{ epoch: 2, loss: 0 }] });
      handler({ type: "status", status: "cancelled" });
      handler({ message: "legacy progress", test: 2 });
      expect(onEvent).toHaveBeenCalledTimes(8);
      expect(onEvent).toHaveBeenLastCalledWith({ message: "legacy progress", test: 2 });

      cleanup();
    });

    it("rejects unsupported, unknown, null, array, and malformed payloads", () => {
      const onEvent = vi.fn();
      subscribeToJob("test-job-123", onEvent);
      const handler = mockListeners.get("result :::")!;
      const envelope = { job_id: "test-job-123", timestamp: "now" };

      handler({ ...envelope, type: "metrics", version: 2, epoch: 1, loss: 0.5 });
      handler({ ...envelope, type: "metrics", epoch: 1, loss: "bad" });
      handler({ ...envelope, type: "metrics", epoch: 1, loss: 0.5, job_id: "other-job" });
      handler({ ...envelope, type: "catchup", version: 1, metrics: [], status: "running" });
      handler({ ...envelope, type: "unknown", version: 1 });
      handler(null);
      handler([]);
      handler({ ...envelope, type: "catchup", version: 1, metrics: {} });
      handler({ ...envelope, type: "status", version: 1 });
      handler({ message: 12, test: 2 });

      expect(onEvent).not.toHaveBeenCalled();
    });
  });

  describe("unsubscribeFromJob", () => {
    it("unsubscribes from job", () => {
      const jobId = "test-job-123";

      // Should not throw
      expect(() => unsubscribeFromJob(jobId)).not.toThrow();
    });

    it("handles null jobId gracefully", () => {
      expect(() => unsubscribeFromJob(null as any)).not.toThrow();
    });
  });

  describe("cancelJob", () => {
    it("sends DELETE request to cancel endpoint", async () => {
      const jobId = "test-job-123";

      await cancelJob(jobId);

      expect(axios.delete).toHaveBeenCalledWith(expect.stringContaining(jobId));
    });

    it("returns promise that resolves", async () => {
      const jobId = "test-job-123";

      const result = await cancelJob(jobId);

      expect(result).toBeDefined();
    });
  });

  describe("getTrainingSocket", () => {
    it("returns a socket instance", () => {
      const socket = getTrainingSocket();
      expect(socket).toBeDefined();
    });
  });
});
