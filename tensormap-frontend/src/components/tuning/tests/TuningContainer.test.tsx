import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TuningContainer from "../TuningContainer";
import { TuningProgress } from "@/services/tuningService";

let progressHandler: ((data: TuningProgress) => void) | null = null;

vi.mock("@/services/tuningService", () => ({
  startTuning: vi.fn(),
  cancelTuning: vi.fn(),
  applyBestParams: vi.fn(),
  getTuningSession: vi.fn(),
  subscribeToTuning: vi.fn((_id, handler) => {
    progressHandler = handler;
    return vi.fn();
  }),
  unsubscribeFromTuning: vi.fn(),
}));

import { startTuning, getTuningSession, subscribeToTuning } from "@/services/tuningService";

describe("TuningContainer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    progressHandler = null;
  });

  it("keeps single TuningProgressPanel mounted through completion and allows resetting for new sweep", async () => {
    const user = userEvent.setup();
    vi.mocked(startTuning).mockResolvedValue({
      data: {
        success: true,
        data: {
          tuning_id: "tune-abc12345",
          status: "pending",
          estimated_seconds: 120,
          n_trials: 1,
        },
      },
    } as any);

    vi.mocked(getTuningSession).mockResolvedValue({
      data: {
        success: true,
        data: {
          tuning_id: "tune-abc12345",
          model_id: 1,
          strategy: "random",
          search_space: {},
          max_trials: 1,
          metric: "val_accuracy",
          direction: "maximize",
          early_stop_threshold: null,
          best_job_id: "job-1",
          best_hyperparams: { optimizer: "adam", lr: 0.001, batch_size: 32, epochs: 10 },
          status: "completed",
          total_trials: 1,
          completed_trials: 1,
          early_stopped: false,
          created_at: null,
          completed_at: null,
          trials: [
            {
              job_id: "job-1",
              status: "completed",
              hyperparams: { optimizer: "adam", lr: 0.001, batch_size: 32, epochs: 10 },
              metric_value: 0.92,
              started_at: null,
              completed_at: null,
            },
          ],
        },
      },
    } as any);

    render(<TuningContainer modelName="iris_net" />);

    await user.click(screen.getByRole("button", { name: /Start Hyperparameter Tuning/i }));

    await waitFor(() => {
      expect(screen.getByText("Tuning Progress")).toBeInTheDocument();
    });
    expect(subscribeToTuning).toHaveBeenCalledTimes(1);

    // Emit trial progress and completion on the single subscriber
    await act(async () => {
      progressHandler?.({
        type: "tuning_progress",
        trial: 1,
        total: 1,
        job_id: "job-1",
        hyperparams: { optimizer: "adam", lr: 0.001, batch_size: 32, epochs: 10 },
        metric: 0.92,
        best_metric: 0.92,
        best_job_id: "job-1",
      });

      progressHandler?.({
        type: "tuning_complete",
        status: "completed",
        best_job_id: "job-1",
        best_metric: 0.92,
      });
    });

    await waitFor(() => {
      expect(screen.getByText("Best Parameters Found")).toBeInTheDocument();
    });
    // Preserved single progress panel subscription (did NOT unmount/remount on completion)
    expect(subscribeToTuning).toHaveBeenCalledTimes(1);
    expect(screen.getByText("val_accuracy: 0.9200")).toBeInTheDocument();

    // Reset button returns to config form
    await user.click(screen.getByRole("button", { name: /Configure New Sweep/i }));
    expect(screen.getByRole("button", { name: /Start Hyperparameter Tuning/i })).toBeInTheDocument();
  });
});
