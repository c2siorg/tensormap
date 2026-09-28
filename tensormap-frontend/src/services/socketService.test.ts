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

    it("passes supported telemetry and legacy listener payloads, and ignores unsupported telemetry versions", () => {
      const jobId = "test-job-123";
      const onEvent = vi.fn();

      const cleanup = subscribeToJob(jobId, onEvent);

      const handler = mockListeners.get("result :::")!;
      handler({ type: "metrics", version: 1, epoch: 1 });
      handler({ message: "legacy progress", test: 2 });
      handler({ type: "metrics", version: 2, epoch: 2 });
      expect(onEvent).toHaveBeenCalledTimes(2);
      expect(onEvent).toHaveBeenLastCalledWith({ message: "legacy progress", test: 2 });

      cleanup();
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
