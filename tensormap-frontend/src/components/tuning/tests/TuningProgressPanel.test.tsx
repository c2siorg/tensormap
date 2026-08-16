/**
 * Test suite for TuningProgressPanel component.
 * @module
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TuningProgressPanel from "../TuningProgressPanel";
import { TuningProgress } from "@/services/tuningService";

// Mock the tuning service
vi.mock("@/services/tuningService", () => ({
  subscribeToTuning: vi.fn(),
  unsubscribeFromTuning: vi.fn(),
}));

import { subscribeToTuning, unsubscribeFromTuning } from "@/services/tuningService";

describe("TuningProgressPanel", () => {
  const mockOnCancel = vi.fn();
  const mockOnComplete = vi.fn();
  let mockProgressHandler: ((data: TuningProgress) => void) | null = null;

  beforeEach(() => {
    mockOnCancel.mockClear();
    mockOnComplete.mockClear();

    // Mock subscribeToTuning to capture the progress handler
    vi.mocked(subscribeToTuning).mockImplementation((tuningId, handler) => {
      mockProgressHandler = handler;
      return vi.fn(); // Return cleanup function
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    mockProgressHandler = null;
  });

  it("shows trial results table", async () => {
    render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    // Simulate catch-up event
    mockProgressHandler?.({
      type: "tuning_catchup",
      completed_trials: 2,
      total_trials: 4,
      status: "running",
      early_stopped: false,
      trials: [
        {
          job_id: "job-1",
          status: "completed",
          hyperparams: { optimizer: "adam", learning_rate: 0.001, batch_size: 32, epochs: 50 },
          metric_value: 0.85,
          started_at: null,
          completed_at: null,
        },
        {
          job_id: "job-2",
          status: "completed",
          hyperparams: { optimizer: "sgd", learning_rate: 0.01, batch_size: 64, epochs: 50 },
          metric_value: 0.82,
          started_at: null,
          completed_at: null,
        },
      ],
    });

    await waitFor(() => {
      // Check table headers
      expect(screen.getByText("Optimizer")).toBeTruthy();
      expect(screen.getByText("Learning Rate")).toBeTruthy();
      expect(screen.getByText("Batch Size")).toBeTruthy();
      expect(screen.getByText("Epochs")).toBeTruthy();
      expect(screen.getByText("val_accuracy")).toBeTruthy();

      // Check trial data
      expect(screen.getByText("adam")).toBeTruthy();
      expect(screen.getByText("sgd")).toBeTruthy();
      expect(screen.getByText("0.8500")).toBeTruthy();
      expect(screen.getByText("0.8200")).toBeTruthy();
    });
  });

  it("Socket.IO progress event appends row to table", async () => {
    const { rerender } = render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    // Start with one trial
    mockProgressHandler?.({
      type: "tuning_progress",
      trial: 1,
      total: 4,
      hyperparams: { optimizer: "adam", learning_rate: 0.001, batch_size: 32, epochs: 50 },
      metric: 0.85,
      best_metric: 0.85,
      best_job_id: "trial-1",
    });

    await waitFor(() => {
      expect(screen.getByText("adam")).toBeTruthy();
    });

    // Add second trial
    mockProgressHandler?.({
      type: "tuning_progress",
      trial: 2,
      total: 4,
      hyperparams: { optimizer: "sgd", learning_rate: 0.01, batch_size: 64, epochs: 50 },
      metric: 0.87,
      best_metric: 0.87,
      best_job_id: "trial-2",
    });

    await waitFor(() => {
      expect(screen.getByText("sgd")).toBeTruthy();
      expect(screen.getByText("0.8700")).toBeTruthy();
    });
  });

  it("best row highlighted when best_job_id changes", async () => {
    const { container } = render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    // Send trials with best
    mockProgressHandler?.({
      type: "tuning_catchup",
      completed_trials: 2,
      total_trials: 4,
      status: "running",
      early_stopped: false,
      trials: [
        {
          job_id: "job-1",
          status: "completed",
          hyperparams: { optimizer: "adam", learning_rate: 0.001, batch_size: 32, epochs: 50 },
          metric_value: 0.85,
          started_at: null,
          completed_at: null,
        },
        {
          job_id: "job-2",
          status: "completed",
          hyperparams: { optimizer: "sgd", learning_rate: 0.01, batch_size: 64, epochs: 50 },
          metric_value: 0.87,
          started_at: null,
          completed_at: null,
        },
      ],
    });

    // Set best job
    mockProgressHandler?.({
      type: "tuning_progress",
      trial: 2,
      total: 4,
      hyperparams: { optimizer: "sgd", learning_rate: 0.01, batch_size: 64, epochs: 50 },
      metric: 0.87,
      best_metric: 0.87,
      best_job_id: "job-2",
    });

    await waitFor(() => {
      // Check for BEST badge
      expect(screen.getByText("BEST")).toBeTruthy();

      // Check for green background on best row
      const rows = container.querySelectorAll("tr.bg-green-50");
      expect(rows.length).toBeGreaterThan(0);
    });
  });

  it("displays early-stop message when applicable", async () => {
    render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    mockProgressHandler?.({
      type: "tuning_complete",
      status: "completed",
      best_metric: 0.95,
      best_job_id: "job-1",
    });

    // Set early stopped flag
    mockProgressHandler?.({
      type: "tuning_catchup",
      completed_trials: 2,
      total_trials: 4,
      status: "completed",
      early_stopped: true,
      trials: [],
    });

    await waitFor(() => {
      expect(screen.getByText(/Early Stopped/i)).toBeTruthy();
      expect(screen.getByText(/target metric reached/i)).toBeTruthy();
    });
  });

  it("cancel button calls onCancel", async () => {
    const user = userEvent.setup();

    render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    // Set status to running
    mockProgressHandler?.({
      type: "tuning_progress",
      trial: 1,
      total: 4,
      hyperparams: {},
      metric: 0.8,
    });

    await waitFor(() => {
      expect(screen.getByText("Running")).toBeTruthy();
    });

    const cancelButton = screen.getByRole("button", { name: /Cancel/i });
    await user.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("shows progress bar with correct percentage", async () => {
    render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    mockProgressHandler?.({
      type: "tuning_progress",
      trial: 2,
      total: 4,
      hyperparams: {},
      metric: 0.8,
    });

    await waitFor(() => {
      expect(screen.getByText("Trial 2 / 4")).toBeTruthy();
      expect(screen.getByText("50%")).toBeTruthy();
    });
  });

  it("calls onComplete when tuning finishes", async () => {
    render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
        onComplete={mockOnComplete}
      />,
    );

    mockProgressHandler?.({
      type: "tuning_complete",
      status: "completed",
      best_metric: 0.95,
      best_job_id: "job-1",
    });

    await waitFor(() => {
      expect(mockOnComplete).toHaveBeenCalledTimes(1);
    });
  });

  it("unsubscribes on unmount", () => {
    const { unmount } = render(
      <TuningProgressPanel
        tuningId="test-tuning-123"
        totalTrials={4}
        metric="val_accuracy"
        direction="maximize"
        onCancel={mockOnCancel}
      />,
    );

    unmount();

    expect(unsubscribeFromTuning).toHaveBeenCalledWith("test-tuning-123");
  });
});
