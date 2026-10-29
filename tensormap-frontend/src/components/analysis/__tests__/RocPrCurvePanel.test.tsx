/**
 * Tests for RocPrCurvePanel component
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RocPrCurvePanel from "../RocPrCurvePanel";
import { RocCurveData } from "@/types/analysis";

describe("RocPrCurvePanel", () => {
  const mockMultiClassData: RocCurveData = {
    classes: [
      {
        class_index: 0,
        class_name: "Setosa",
        roc: {
          fpr: [0.0, 0.1, 0.5, 1.0],
          tpr: [0.0, 0.8, 0.95, 1.0],
          thresholds: [1.8, 0.9, 0.5, 0.1],
          auc: 0.925,
        },
        pr: {
          precision: [1.0, 0.9, 0.85, 0.5],
          recall: [0.0, 0.8, 0.95, 1.0],
          thresholds: [0.9, 0.5, 0.1],
          auc: 0.895,
        },
      },
      {
        class_index: 1,
        class_name: "Versicolor",
        roc: {
          fpr: [0.0, 0.2, 0.4, 1.0],
          tpr: [0.0, 0.7, 0.88, 1.0],
          thresholds: [1.7, 0.8, 0.4, 0.1],
          auc: 0.865,
        },
        pr: {
          precision: [1.0, 0.85, 0.75, 0.5],
          recall: [0.0, 0.7, 0.88, 1.0],
          thresholds: [0.8, 0.4, 0.1],
          auc: 0.825,
        },
      },
    ],
    macro_roc_auc: 0.895,
    macro_pr_auc: 0.860,
    is_binary: false,
    class_names: ["Setosa", "Versicolor"],
    n_samples: 100,
    analysis_type: "classification",
    cached: false,
  };

  const mockBinaryData: RocCurveData = {
    classes: [
      {
        class_index: 1,
        class_name: "Positive",
        roc: {
          fpr: [0.0, 0.05, 0.2, 1.0],
          tpr: [0.0, 0.85, 0.98, 1.0],
          thresholds: [1.5, 0.9, 0.5, 0.0],
          auc: 0.965,
        },
        pr: {
          precision: [1.0, 0.95, 0.88, 0.5],
          recall: [0.0, 0.85, 0.98, 1.0],
          thresholds: [0.9, 0.5, 0.0],
          auc: 0.942,
        },
      },
    ],
    macro_roc_auc: 0.965,
    macro_pr_auc: 0.942,
    is_binary: true,
    class_names: ["Negative", "Positive"],
    n_samples: 50,
    analysis_type: "classification",
    cached: true,
  };

  it("renders panel title, subtitle and ROC mode by default", () => {
    render(<RocPrCurvePanel data={mockMultiClassData} />);

    expect(screen.getByText("ROC & Precision-Recall Curves")).toBeInTheDocument();
    expect(
      screen.getByText("Threshold-varying diagnostic performance across decision boundaries")
    ).toBeInTheDocument();
    expect(screen.getByText(/Macro ROC-AUC:/)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "ROC Curve" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Precision-Recall" })).toBeInTheDocument();
  });

  it("switches to Precision-Recall mode when clicking tab", async () => {
    const user = userEvent.setup();
    render(<RocPrCurvePanel data={mockMultiClassData} />);

    const prTab = screen.getByRole("tab", { name: "Precision-Recall" });
    await user.click(prTab);

    expect(await screen.findByText(/Macro PR-AUC:/)).toBeInTheDocument();
    expect(screen.queryByText(/Macro ROC-AUC:/)).not.toBeInTheDocument();
  });

  it("displays multi-class OvR selector and allows class filtering", () => {
    render(<RocPrCurvePanel data={mockMultiClassData} />);

    expect(screen.getByText("Class View:")).toBeInTheDocument();
  });

  it("renders binary classification data without multi-class class selector", () => {
    render(<RocPrCurvePanel data={mockBinaryData} />);

    expect(screen.getByText(/Macro ROC-AUC:/)).toBeInTheDocument();
    expect(screen.queryByText("Class View:")).not.toBeInTheDocument();
  });

  it("renders loading state spinner", () => {
    render(<RocPrCurvePanel data={null} isLoading={true} />);

    expect(screen.getByText("Loading curve analysis...")).toBeInTheDocument();
  });

  it("renders error message alert when error prop is passed", () => {
    render(<RocPrCurvePanel data={null} error="Failed to compute curves" />);

    expect(screen.getByText("Evaluation Error:")).toBeInTheDocument();
    expect(screen.getByText("Failed to compute curves")).toBeInTheDocument();
  });

  it("renders empty state fallback when data is null", () => {
    render(<RocPrCurvePanel data={null} />);

    expect(
      screen.getByText("No curve evaluation data available for this model.")
    ).toBeInTheDocument();
  });
});
