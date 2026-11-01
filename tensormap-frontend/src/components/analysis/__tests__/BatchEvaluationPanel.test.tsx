import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import BatchEvaluationPanel from "../BatchEvaluationPanel";
import * as FileServices from "../../../services/FileServices";
import * as EvaluationService from "../../../services/EvaluationService";

describe("BatchEvaluationPanel", () => {
  const mockJobId = "job-eval-123";
  const mockFiles = [
    { file_id: "file-1", file_name: "test_holdout.csv", file_type: "csv" },
    { file_id: "file-2", file_name: "validation.csv", file_type: "csv" },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(FileServices, "getAllFiles").mockResolvedValue(mockFiles as any);
  });

  it("renders panel title, description, and populates dataset dropdown", async () => {
    render(<BatchEvaluationPanel jobId={mockJobId} problemType="classification" />);

    expect(screen.getByText("Holdout Dataset Evaluation & Batch Scoring")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: /select test dataset/i })).toBeInTheDocument();
    });

    const select = screen.getByRole("combobox", {
      name: /select test dataset/i,
    }) as HTMLSelectElement;
    expect(select.options.length).toBe(2);
    expect(select.options[0].text).toBe("test_holdout.csv");
    expect(select.options[1].text).toBe("validation.csv");
  });

  it("shows empty state when no CSV files are available", async () => {
    vi.spyOn(FileServices, "getAllFiles").mockResolvedValue([]);

    render(<BatchEvaluationPanel jobId={mockJobId} problemType="classification" />);

    await waitFor(() => {
      expect(screen.getByText("No CSV datasets found")).toBeInTheDocument();
    });

    const evalButton = screen.getByRole("button", { name: /evaluate dataset/i });
    expect(evalButton).toBeDisabled();
  });

  it("runs classification evaluation and displays scorecard metrics and top errors", async () => {
    const mockEvalResponse = {
      job_id: mockJobId,
      file_id: "file-1",
      problem_type: "classification" as const,
      total_samples: 50,
      evaluated_at: new Date().toISOString(),
      metrics: {
        accuracy: 0.92,
        f1_macro: 0.915,
        f1_weighted: 0.921,
        precision_macro: 0.93,
        log_loss: 0.2345,
      },
      top_errors: [
        {
          index: 4,
          features: { sepal_length: 5.1, sepal_width: 3.5 },
          actual_label: "setosa",
          predicted_label: "versicolor",
          confidence: 0.88,
          loss_or_residual: 0.88,
        },
      ],
    };

    vi.spyOn(EvaluationService, "evaluateHoldoutDataset").mockResolvedValue(mockEvalResponse);

    render(<BatchEvaluationPanel jobId={mockJobId} problemType="classification" />);

    await waitFor(() => {
      expect(screen.getByText("test_holdout.csv")).toBeInTheDocument();
    });

    const evalButton = screen.getByRole("button", { name: /evaluate dataset/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(screen.getByText("92.0%")).toBeInTheDocument();
    });

    expect(screen.getByText("0.9150")).toBeInTheDocument();
    expect(screen.getByText("0.9210")).toBeInTheDocument();
    expect(screen.getByText("0.9300")).toBeInTheDocument();
    expect(screen.getByText("0.2345")).toBeInTheDocument();

    // Verify top error sample
    expect(screen.getByText("Top Prediction Errors (1)")).toBeInTheDocument();
    expect(screen.getByText("#4")).toBeInTheDocument();
    expect(screen.getByText("setosa")).toBeInTheDocument();
    expect(screen.getByText("versicolor")).toBeInTheDocument();
    expect(screen.getByText("88.0%")).toBeInTheDocument();
  });

  it("runs regression evaluation and displays RMSE, MAE, R2, and residuals", async () => {
    const mockRegResponse = {
      job_id: mockJobId,
      file_id: "file-1",
      problem_type: "regression" as const,
      total_samples: 40,
      evaluated_at: new Date().toISOString(),
      metrics: {
        rmse: 12.3456,
        mae: 8.7654,
        r2: 0.8543,
        mse: 152.4138,
        max_error: 25.12,
      },
      top_errors: [
        {
          index: 12,
          features: { size: 120.0, rooms: 3.0 },
          actual_label: "300.0",
          predicted_label: "274.88",
          confidence: null,
          loss_or_residual: 25.12,
        },
      ],
    };

    vi.spyOn(EvaluationService, "evaluateHoldoutDataset").mockResolvedValue(mockRegResponse);

    render(<BatchEvaluationPanel jobId={mockJobId} problemType="regression" />);

    await waitFor(() => {
      expect(screen.getByText("test_holdout.csv")).toBeInTheDocument();
    });

    const evalButton = screen.getByRole("button", { name: /evaluate dataset/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(screen.getByText("12.3456")).toBeInTheDocument();
    });

    expect(screen.getByText("8.7654")).toBeInTheDocument();
    expect(screen.getByText("0.8543")).toBeInTheDocument();
    expect(screen.getByText("152.4138")).toBeInTheDocument();
    expect(screen.getAllByText("25.1200").length).toBeGreaterThanOrEqual(1);

    // Verify error inspector
    expect(screen.getByText("#12")).toBeInTheDocument();
    expect(screen.getByText("300.0")).toBeInTheDocument();
    expect(screen.getByText("274.88")).toBeInTheDocument();
  });

  it("displays error alert when evaluation API call fails", async () => {
    vi.spyOn(EvaluationService, "evaluateHoldoutDataset").mockRejectedValue({
      response: {
        data: { detail: "Evaluation dataset is missing required feature columns: feat_x" },
      },
    });

    render(<BatchEvaluationPanel jobId={mockJobId} problemType="classification" />);

    await waitFor(() => {
      expect(screen.getByText("test_holdout.csv")).toBeInTheDocument();
    });

    const evalButton = screen.getByRole("button", { name: /evaluate dataset/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(
        screen.getByText(/Evaluation dataset is missing required feature columns: feat_x/i),
      ).toBeInTheDocument();
    });
  });

  it("triggers batch predictions CSV download when Scored CSV button is clicked", async () => {
    const downloadSpy = vi
      .spyOn(EvaluationService, "downloadBatchPredictionsCsv")
      .mockResolvedValue();

    render(<BatchEvaluationPanel jobId={mockJobId} problemType="classification" />);

    await waitFor(() => {
      expect(screen.getByText("test_holdout.csv")).toBeInTheDocument();
    });

    const exportBtn = screen.getByTitle("Download batch predictions as CSV");
    fireEvent.click(exportBtn);

    await waitFor(() => {
      expect(downloadSpy).toHaveBeenCalledWith(mockJobId, "file-1");
    });
  });
});
