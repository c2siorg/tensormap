import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import FeatureSelectionStudio from "../FeatureSelectionStudio";
import * as featureService from "../../../services/FeatureSelectionService";
import * as fileServices from "../../../services/FileServices";

vi.mock("../../../services/FeatureSelectionService", () => ({
  analyzeFeatureRelevance: vi.fn(),
  applyFeatureSelection: vi.fn(),
}));

vi.mock("../../../services/FileServices", () => ({
  getCovMatrix: vi.fn(),
}));

const MOCK_COLUMNS_RESPONSE = {
  data_types: {
    age: "float64",
    salary: "float64",
    noise: "float64",
    const_col: "float64",
    churn: "int64",
  },
};

const MOCK_ANALYSIS_RESPONSE = {
  file_id: "file-123",
  target_field: "churn",
  problem_type: "classification",
  total_samples: 100,
  total_features: 4,
  critical_features_count: 1,
  informative_features_count: 1,
  weak_features_count: 1,
  constant_features_count: 1,
  features: [
    {
      feature: "age",
      mutual_info: 0.4521,
      f_statistic: 24.51,
      p_value: 0.00001,
      variance: 15.2,
      normalized_score: 92.5,
      tier: "CRITICAL",
      is_constant: false,
      suggested_action: "keep",
    },
    {
      feature: "salary",
      mutual_info: 0.1845,
      f_statistic: 6.82,
      p_value: 0.012,
      variance: 52000.0,
      normalized_score: 42.0,
      tier: "INFORMATIVE",
      is_constant: false,
      suggested_action: "keep",
    },
    {
      feature: "noise",
      mutual_info: 0.021,
      f_statistic: 0.95,
      p_value: 0.35,
      variance: 18.0,
      normalized_score: 8.5,
      tier: "WEAK",
      is_constant: false,
      suggested_action: "review",
    },
    {
      feature: "const_col",
      mutual_info: 0.0,
      f_statistic: null,
      p_value: null,
      variance: 0.0,
      normalized_score: 0.0,
      tier: "CONSTANT_OR_REDUNDANT",
      is_constant: true,
      suggested_action: "prune",
    },
  ],
};

describe("FeatureSelectionStudio Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fileServices.getCovMatrix.mockResolvedValue(MOCK_COLUMNS_RESPONSE);
  });

  it("renders target selector, title, and analyze button", async () => {
    render(<FeatureSelectionStudio fileId="file-123" />);

    expect(screen.getByText("Feature Relevance & Selection Studio")).toBeTruthy();
    expect(screen.getByText("Target Column:")).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText("Analyze Relevance")).toBeTruthy();
      const options = screen.getAllByRole("option");
      expect(options.length).toBeGreaterThanOrEqual(5);
    });
  });

  it("executes analysis and renders stat cards and ranked feature table", async () => {
    const user = userEvent.setup();
    featureService.analyzeFeatureRelevance.mockResolvedValue(MOCK_ANALYSIS_RESPONSE);

    render(<FeatureSelectionStudio fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByText("Analyze Relevance")).toBeTruthy();
    });

    const analyzeBtn = screen.getByText("Analyze Relevance");
    await user.click(analyzeBtn);

    await waitFor(() => {
      expect(screen.getByText("Candidate Features")).toBeTruthy();
      expect(screen.getByText("Critical & Informative")).toBeTruthy();
      expect(screen.getByText("Weak Candidates")).toBeTruthy();
      expect(screen.getByText("Constant / Redundant")).toBeTruthy();
    });

    // Check table rows
    expect(screen.getAllByText("age").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("salary").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("noise").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("const_col").length).toBeGreaterThanOrEqual(1);

    // Check tier badges
    expect(screen.getByText("Critical")).toBeTruthy();
    expect(screen.getByText("Informative")).toBeTruthy();
    expect(screen.getByText("Weak")).toBeTruthy();
    expect(screen.getByText("Redundant / Constant")).toBeTruthy();
  });

  it("supports preset filters and updates feature selection counts", async () => {
    const user = userEvent.setup();
    featureService.analyzeFeatureRelevance.mockResolvedValue(MOCK_ANALYSIS_RESPONSE);

    render(<FeatureSelectionStudio fileId="file-123" />);
    await waitFor(() => screen.getByText("Analyze Relevance"));
    await user.click(screen.getByText("Analyze Relevance"));

    await waitFor(() => {
      expect(screen.getByText("Presets:")).toBeTruthy();
    });

    // Click Top 3 preset
    const top3Btn = screen.getByText("Top 3");
    await user.click(top3Btn);

    await waitFor(() => {
      expect(screen.getByTestId("selection-summary").textContent).toContain("3 of 4 selected");
    });

    // Click Informative Only preset
    const infoOnlyBtn = screen.getByText("Informative Only");
    await user.click(infoOnlyBtn);

    await waitFor(() => {
      expect(screen.getByTestId("selection-summary").textContent).toContain("2 of 4 selected");
    });

    // Click Clear All
    const clearBtn = screen.getByText("Clear");
    await user.click(clearBtn);

    await waitFor(() => {
      expect(screen.getByTestId("selection-summary").textContent).toContain("0 of 4 selected");
    });
  });

  it("opens confirmation dialog and applies dataset pruning", async () => {
    const user = userEvent.setup();
    const mockOnPruned = vi.fn();
    featureService.analyzeFeatureRelevance.mockResolvedValue(MOCK_ANALYSIS_RESPONSE);
    featureService.applyFeatureSelection.mockResolvedValue({
      file_id: "file-123",
      target_field: "churn",
      retained_features: ["age", "salary"],
      dropped_features: ["noise", "const_col"],
      total_columns_remaining: 3,
      backup_created: true,
      message: "Successfully pruned dataset: retained 2 features, dropped 2 features.",
    });

    render(<FeatureSelectionStudio fileId="file-123" onPruned={mockOnPruned} />);
    await waitFor(() => screen.getByText("Analyze Relevance"));
    await user.click(screen.getByText("Analyze Relevance"));

    await waitFor(() => screen.getByText("Prune Dataset"));
    const pruneBtn = screen.getByText("Prune Dataset");
    await user.click(pruneBtn);

    // Verify modal is open
    expect(screen.getByText("Confirm Dataset Pruning")).toBeTruthy();
    expect(screen.getByTestId("backup-note").textContent).toContain("backup file will be created");

    // Click Confirm & Prune
    const confirmBtn = screen.getByText("Confirm & Prune");
    await user.click(confirmBtn);

    await waitFor(() => {
      expect(featureService.applyFeatureSelection).toHaveBeenCalledWith(
        "file-123",
        expect.objectContaining({
          selected_features: expect.any(Array),
        }),
      );
      expect(mockOnPruned).toHaveBeenCalledTimes(1);
    });
  });

  it("displays error banner when backend analysis fails", async () => {
    const user = userEvent.setup();
    featureService.analyzeFeatureRelevance.mockRejectedValue(
      new Error("Target column contains only missing values"),
    );

    render(<FeatureSelectionStudio fileId="file-123" />);
    await waitFor(() => screen.getByText("Analyze Relevance"));
    await user.click(screen.getByText("Analyze Relevance"));

    await waitFor(() => {
      expect(screen.getByText("Target column contains only missing values")).toBeTruthy();
    });
  });
});
