import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ComparisonDashboard from "../ComparisonDashboard";

vi.mock("@/services/comparisonService", () => ({
  getTrainingJobs: vi.fn(),
  compareJobs: vi.fn(),
}));

import { getTrainingJobs, compareJobs } from "@/services/comparisonService";

describe("ComparisonDashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads training jobs and renders overlay comparison when jobs are selected", async () => {
    const user = userEvent.setup();
    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: {
        success: true,
        data: [
          {
            job_id: "job-aaa11111",
            model_id: 1,
            status: "completed",
            started_at: "2026-11-24T10:00:00Z",
            completed_at: "2026-11-24T10:05:00Z",
          },
          {
            job_id: "job-bbb22222",
            model_id: 1,
            status: "completed",
            started_at: "2026-11-24T11:00:00Z",
            completed_at: "2026-11-24T11:05:00Z",
          },
        ],
      },
    } as any);

    vi.mocked(compareJobs).mockResolvedValue({
      data: {
        success: true,
        data: {
          metric: "val_loss",
          jobs: [
            {
              job_id: "job-aaa11111",
              hyperparams: { optimizer: "adam", lr: 0.001, batch_size: 32, epochs: 10 },
              status: "completed",
              started_at: "2026-11-24T10:00:00Z",
              completed_at: "2026-11-24T10:05:00Z",
              metrics: [{ epoch: 1, loss: 0.45, accuracy: 0.8 }],
            },
            {
              job_id: "job-bbb22222",
              hyperparams: { optimizer: "sgd", lr: 0.01, batch_size: 64, epochs: 10 },
              status: "completed",
              started_at: "2026-11-24T11:00:00Z",
              completed_at: "2026-11-24T11:05:00Z",
              metrics: [{ epoch: 1, loss: 0.0, accuracy: 1.0 }],
            },
          ],
        },
      },
    } as any);

    render(<ComparisonDashboard modelName="iris_net" />);

    await waitFor(() => {
      expect(screen.getByText(/job-aaa1/)).toBeInTheDocument();
      expect(screen.getByText(/job-bbb2/)).toBeInTheDocument();
    });

    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]);
    await user.click(checkboxes[1]);

    await waitFor(() => {
      expect(compareJobs).toHaveBeenCalledWith(["job-aaa11111", "job-bbb22222"], "val_loss");
      expect(screen.getByText("Comparison Summary")).toBeInTheDocument();
      expect(screen.getByText("BEST")).toBeInTheDocument();
      // Loss of 0.0000 preserved cleanly without falsy fallback
      expect(screen.getByText("0.0000")).toBeInTheDocument();
      expect(screen.getByText("100.00%")).toBeInTheDocument();
    });
  });

  it("resets selected jobs when modelName changes", async () => {
    const user = userEvent.setup();
    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: {
        success: true,
        data: [
          {
            job_id: "job-aaa11111",
            model_id: 1,
            status: "completed",
            started_at: "2026-11-24T10:00:00Z",
            completed_at: "2026-11-24T10:05:00Z",
          },
        ],
      },
    } as any);

    vi.mocked(compareJobs).mockResolvedValue({
      data: {
        success: true,
        data: {
          metric: "val_loss",
          jobs: [
            {
              job_id: "job-aaa11111",
              hyperparams: { optimizer: "adam" },
              status: "completed",
              started_at: null,
              completed_at: null,
              metrics: [{ epoch: 1, val_loss: 0.2, val_accuracy: 0.9 }],
            },
          ],
        },
      },
    } as any);

    const { rerender } = render(<ComparisonDashboard modelName="model_a" />);

    await waitFor(() => {
      expect(screen.getByRole("checkbox")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("checkbox"));
    await waitFor(() => {
      expect(screen.getByText("Comparison Summary")).toBeInTheDocument();
    });

    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: { success: true, data: [] },
    } as any);

    rerender(<ComparisonDashboard modelName="model_b" />);

    await waitFor(() => {
      expect(screen.getByText("Select training runs above to compare their metrics")).toBeInTheDocument();
    });
    expect(screen.queryByText("Comparison Summary")).not.toBeInTheDocument();
  });

  it("prevents selecting more than 5 jobs", async () => {
    const user = userEvent.setup();
    const sixJobs = Array.from({ length: 6 }, (_, i) => ({
      job_id: `job-0000000${i}`,
      model_id: 1,
      status: "completed",
      started_at: null,
      completed_at: null,
    }));

    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: { success: true, data: sixJobs },
    } as any);
    vi.mocked(compareJobs).mockResolvedValue({
      data: { success: true, data: { metric: "val_loss", jobs: [] } },
    } as any);

    render(<ComparisonDashboard modelName="model_a" />);

    await waitFor(() => {
      expect(screen.getAllByRole("checkbox")).toHaveLength(6);
    });

    const checkboxes = screen.getAllByRole("checkbox");
    for (let i = 0; i < 6; i++) {
      await user.click(checkboxes[i]);
    }

    expect(screen.getByText("Maximum 5 jobs can be compared at once")).toBeInTheDocument();
  });
});
