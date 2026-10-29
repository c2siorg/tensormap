/**
 * Tests for AnalysisDashboard component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import AnalysisDashboard from "../AnalysisDashboard";
import axios from "@/shared/Axios";

vi.mock("@/shared/Axios");
vi.mock("@/hooks/useFeatureImportancePoller", () => ({
  useFeatureImportancePoller: () => ({
    data: null,
    isLoading: false,
    error: null,
  }),
}));
vi.mock("../PredictionExplorer", () => ({ default: () => null }));

describe("AnalysisDashboard", () => {
  const mockClassificationResponse = {
    data: {
      confusion_matrix: [
        [10, 2],
        [1, 15],
      ],
      classification_report: {
        "Class A": {
          precision: 0.91,
          recall: 0.83,
          "f1-score": 0.87,
          support: 12,
        },
        "Class B": {
          precision: 0.88,
          recall: 0.94,
          "f1-score": 0.91,
          support: 16,
        },
        "macro avg": {
          precision: 0.9,
          recall: 0.88,
          "f1-score": 0.89,
          support: 28,
        },
        "weighted avg": {
          precision: 0.9,
          recall: 0.89,
          "f1-score": 0.89,
          support: 28,
        },
        accuracy: 0.89,
      },
      class_names: ["Class A", "Class B"],
      overall_accuracy: 0.89,
      n_samples: 28,
      analysis_type: "classification",
      cached: false,
    },
  };

  const mockRegressionResponse = {
    data: {
      y_pred: [1.2, 3.4, 5.6],
      y_true: [1.0, 3.5, 5.5],
      residuals: [0.2, -0.1, 0.1],
      mae: 0.133,
      mse: 0.02,
      analysis_type: "regression",
      cached: false,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:test"),
    });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  });

  it("mounts correct panels for classification type", async () => {
    (axios.get as any).mockResolvedValue(mockClassificationResponse);

    render(<AnalysisDashboard jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Confusion Matrix")).toBeInTheDocument();
      expect(screen.getByText("Classification Report")).toBeInTheDocument();
    });
  });

  it("mounts regression panel for regression type", async () => {
    // First call fails (not classification), second succeeds (regression)
    (axios.get as any)
      .mockRejectedValueOnce({ response: { status: 400 } })
      .mockResolvedValueOnce(mockRegressionResponse);

    render(<AnalysisDashboard jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Predicted vs Actual/)).toBeInTheDocument();
    });
  });

  it("shows loading state initially", () => {
    (axios.get as any).mockImplementation(
      () => new Promise(() => {}), // Never resolves
    );

    render(<AnalysisDashboard jobId="test-job-123" />);

    expect(screen.getByText(/Loading analysis/)).toBeInTheDocument();
  });

  it("handles error state", async () => {
    (axios.get as any).mockRejectedValue({
      response: { status: 500, data: { detail: "Server error" } },
    });

    render(<AnalysisDashboard jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Server error/)).toBeInTheDocument();
    });
  });

  it("selects a format and downloads analysis for the displayed job", async () => {
    const blob = new Blob(["analysis"]);
    (axios.get as any)
      .mockResolvedValueOnce(mockClassificationResponse)
      .mockResolvedValueOnce({ data: blob });
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:test"),
    });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<AnalysisDashboard jobId="job-500" />);
    const format = await screen.findByRole("combobox", { name: "Export format" });
    fireEvent.change(format, { target: { value: "csv" } });
    fireEvent.click(screen.getByRole("button", { name: "Export analysis" }));

    await waitFor(() =>
      expect(axios.get).toHaveBeenNthCalledWith(2, "/model/analysis/job-500/export", {
        params: { format: "csv" },
        responseType: "blob",
      }),
    );
    expect(click).toHaveBeenCalledOnce();
    expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test");
    click.mockRestore();
  });

  it("shows export failures and disables duplicate requests while loading", async () => {
    let finishExport: ((value: unknown) => void) | undefined;
    (axios.get as any).mockResolvedValueOnce(mockClassificationResponse).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishExport = resolve;
        }),
    );
    render(<AnalysisDashboard jobId="job-500" />);

    const button = await screen.findByRole("button", { name: "Export analysis" });
    fireEvent.click(button);
    expect(await screen.findByRole("button", { name: "Exporting…" })).toBeDisabled();
    expect(axios.get).toHaveBeenCalledTimes(2);
    finishExport?.({ data: new Blob(["ok"]) });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Export analysis" })).toBeEnabled(),
    );

    (axios.get as any).mockRejectedValueOnce({ message: "Export failed" });
    fireEvent.click(screen.getByRole("button", { name: "Export analysis" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Export failed");
  });
});
