/**
 * Tests for AnalysisDashboard component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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
          precision: 0.90,
          recall: 0.88,
          "f1-score": 0.89,
          support: 28,
        },
        "weighted avg": {
          precision: 0.90,
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

  const mockRocCurveResponse = {
    data: {
      classes: [
        {
          class_index: 0,
          class_name: "Class A",
          roc: {
            fpr: [0.0, 0.1, 1.0],
            tpr: [0.0, 0.9, 1.0],
            thresholds: [1.5, 0.5, 0.0],
            auc: 0.95,
          },
          pr: {
            precision: [1.0, 0.9, 0.5],
            recall: [0.0, 0.9, 1.0],
            thresholds: [0.9, 0.5, 0.0],
            auc: 0.92,
          },
        },
      ],
      macro_roc_auc: 0.95,
      macro_pr_auc: 0.92,
      is_binary: true,
      class_names: ["Class A"],
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
      mse: 0.020,
      analysis_type: "regression",
      cached: false,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("mounts correct panels for classification type", async () => {
    (axios.get as any).mockImplementation((url: string) => {
      if (url.includes("roc-curve")) {
        return Promise.resolve(mockRocCurveResponse);
      }
      return Promise.resolve(mockClassificationResponse);
    });

    render(<AnalysisDashboard jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Confusion Matrix")).toBeInTheDocument();
      expect(screen.getByText("Classification Report")).toBeInTheDocument();
      expect(screen.getByText("ROC & Precision-Recall Curves")).toBeInTheDocument();
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
});
