import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Analysis from "../Analysis";

vi.mock("@/services/ModelServices", () => ({
  getTrainingHistory: vi.fn(),
}));

vi.mock("@/services/comparisonService", () => ({
  getTrainingJobs: vi.fn(),
}));

vi.mock("@/components/analysis/AnalysisDashboard", () => ({
  default: ({ jobId }) => <div data-testid="analysis-dashboard">AnalysisDashboard: {jobId}</div>,
}));

vi.mock("@/components/comparison", () => ({
  ComparisonDashboard: ({ modelName }) => (
    <div data-testid="comparison-dashboard">ComparisonDashboard: {modelName}</div>
  ),
}));

vi.mock("@/components/export/ExportPanel", () => ({
  default: ({ jobId, modelName }) => (
    <div data-testid="export-panel">
      ExportPanel: {jobId} ({modelName})
    </div>
  ),
}));

vi.mock("@/components/training/ReproducibilityDetails", () => ({
  default: ({ jobId }) => (
    <div data-testid="reproducibility-details">ReproducibilityDetails: {jobId}</div>
  ),
}));

import { getTrainingHistory } from "@/services/ModelServices";
import { getTrainingJobs } from "@/services/comparisonService";

function renderAnalysis(initialUrl = "/workspace/proj-1/analysis") {
  return render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <Routes>
        <Route path="/workspace/:projectId/analysis" element={<Analysis />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Analysis & Compare Container Page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders empty state when project has no models", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([]);

    renderAnalysis();

    await waitFor(() => {
      expect(screen.getByText(/No models found in this project/i)).toBeInTheDocument();
    });
  });

  it("auto-selects completed run, mounts AnalysisDashboard, Export/Reproducibility, and Comparison tabs", async () => {
    const user = userEvent.setup();
    vi.mocked(getTrainingHistory).mockResolvedValue([
      { id: 1, model_name: "iris_clf", epochs: 10, training_split: 80 },
    ]);
    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: {
        success: true,
        data: [
          {
            job_id: "job-running-1",
            model_id: 1,
            status: "running",
            started_at: "2026-11-24T10:00:00Z",
            completed_at: null,
          },
          {
            job_id: "job-done-2",
            model_id: 1,
            status: "completed",
            started_at: "2026-11-23T10:00:00Z",
            completed_at: "2026-11-23T10:02:00Z",
          },
        ],
      },
    });

    renderAnalysis();

    await waitFor(() => {
      expect(screen.getByTestId("analysis-dashboard")).toHaveTextContent(
        "AnalysisDashboard: job-done-2",
      );
    });

    await user.click(screen.getByRole("tab", { name: /Export & Reproducibility/i }));
    await waitFor(() => {
      expect(screen.getByTestId("export-panel")).toHaveTextContent(
        "ExportPanel: job-done-2 (iris_clf)",
      );
      expect(screen.getByTestId("reproducibility-details")).toHaveTextContent(
        "ReproducibilityDetails: job-done-2",
      );
    });

    await user.click(screen.getByRole("tab", { name: /Compare Runs/i }));
    await waitFor(() => {
      expect(screen.getByTestId("comparison-dashboard")).toHaveTextContent(
        "ComparisonDashboard: iris_clf",
      );
    });
  });

  it("shows non-completed notice when selected ?jobId is still running", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([
      { id: 1, model_name: "iris_clf", epochs: 10, training_split: 80 },
    ]);
    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: {
        success: true,
        data: [
          {
            job_id: "job-running-1",
            model_id: 1,
            status: "running",
            started_at: "2026-11-24T10:00:00Z",
            completed_at: null,
          },
        ],
      },
    });

    renderAnalysis("/workspace/proj-1/analysis?model=iris_clf&jobId=job-running-1");

    await waitFor(() => {
      expect(screen.getByText(/Run not completed/i)).toBeInTheDocument();
    });
    expect(screen.queryByTestId("analysis-dashboard")).not.toBeInTheDocument();
  });

  it("shows empty runs message when model has no training jobs", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([
      { id: 1, model_name: "iris_clf", epochs: 10, training_split: 80 },
    ]);
    vi.mocked(getTrainingJobs).mockResolvedValue({
      data: { success: true, data: [] },
    });

    renderAnalysis();

    await waitFor(() => {
      expect(screen.getByText(/No training runs found for/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Go to Training/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Go to Tuning/i })).toBeInTheDocument();
    });
  });

  it("changing run or model updates views without reloading models or unmounting", async () => {
    const user = userEvent.setup();
    vi.mocked(getTrainingHistory).mockResolvedValue([
      { id: 1, model_name: "iris_clf", epochs: 10, training_split: 80 },
      { id: 2, model_name: "mnist_cnn", epochs: 5, training_split: 80 },
    ]);
    vi.mocked(getTrainingJobs).mockImplementation(async (modelName) => {
      if (modelName === "iris_clf") {
        return {
          data: {
            success: true,
            data: [
              { job_id: "job-1111", model_id: 1, status: "completed", started_at: null },
              { job_id: "job-2222", model_id: 1, status: "completed", started_at: null },
            ],
          },
        };
      }
      return {
        data: {
          success: true,
          data: [{ job_id: "job-3333", model_id: 2, status: "completed", started_at: null }],
        },
      };
    });

    renderAnalysis();

    await waitFor(() => {
      expect(screen.getByTestId("analysis-dashboard")).toHaveTextContent(
        "AnalysisDashboard: job-1111",
      );
    });
    expect(getTrainingHistory).toHaveBeenCalledTimes(1);
    expect(getTrainingJobs).toHaveBeenCalledTimes(1);

    // Switch training run via selector (updates searchParams without refetching models or jobs)
    await user.click(screen.getByRole("combobox", { name: /Select training run/i }));
    await user.click(await screen.findByRole("option", { name: /job-2222/i }));

    await waitFor(() => {
      expect(screen.getByTestId("analysis-dashboard")).toHaveTextContent(
        "AnalysisDashboard: job-2222",
      );
    });
    expect(screen.queryByText(/Loading models\.\.\./i)).not.toBeInTheDocument();
    expect(getTrainingHistory).toHaveBeenCalledTimes(1);
    expect(getTrainingJobs).toHaveBeenCalledTimes(1);

    // Switch model via selector (fetches jobs once for new model, never refetches models)
    await user.click(screen.getByRole("combobox", { name: /Select model for analysis/i }));
    await user.click(await screen.findByRole("option", { name: "mnist_cnn" }));

    await waitFor(() => {
      expect(screen.getByTestId("analysis-dashboard")).toHaveTextContent(
        "AnalysisDashboard: job-3333",
      );
    });
    expect(screen.queryByText(/Loading models\.\.\./i)).not.toBeInTheDocument();
    expect(getTrainingHistory).toHaveBeenCalledTimes(1);
    expect(getTrainingJobs).toHaveBeenCalledWith("mnist_cnn");
  });
});
