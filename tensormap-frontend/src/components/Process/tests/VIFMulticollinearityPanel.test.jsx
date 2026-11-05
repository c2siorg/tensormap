import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import VIFMulticollinearityPanel from "../VIFMulticollinearityPanel";
import * as FileServices from "../../../services/FileServices";

vi.mock("../../../services/FileServices", () => ({
  getVIFMetrics: vi.fn(),
}));

const mockSuccessData = {
  vif_scores: [
    {
      feature: "sqft_living",
      vif: 12.45,
      vif_numeric: 12.45,
      tolerance: 0.0803,
      r_squared: 0.9197,
      severity: "HIGH",
      recommendation:
        "High collinearity detected. Consider dropping, combining, or applying L1/L2 regularization.",
    },
    {
      feature: "sqft_above",
      vif: 7.2,
      vif_numeric: 7.2,
      tolerance: 0.1389,
      r_squared: 0.8611,
      severity: "MODERATE",
      recommendation: "Moderate collinearity. Monitor performance or feature importance variance.",
    },
    {
      feature: "bedrooms",
      vif: 2.15,
      vif_numeric: 2.15,
      tolerance: 0.4651,
      r_squared: 0.5349,
      severity: "LOW",
      recommendation: "Healthy feature with low multicollinearity.",
    },
    {
      feature: "sqft_duplicate",
      vif: "Infinity",
      vif_numeric: 99999.0,
      tolerance: 0.0,
      r_squared: 1.0,
      severity: "PERFECT_COLLINEARITY",
      recommendation:
        "High collinearity detected. Consider dropping, combining, or applying L1/L2 regularization.",
    },
  ],
  constant_columns: ["dummy_constant"],
  summary: {
    overall_multicollinearity: "HIGH",
    condition_index: 34.2,
    total_features_evaluated: 4,
    high_vif_features: 2,
    moderate_vif_features: 1,
    low_vif_features: 1,
  },
};

describe("VIFMulticollinearityPanel Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state initially while fetching", async () => {
    FileServices.getVIFMetrics.mockImplementation(() => new Promise(() => {}));
    render(<VIFMulticollinearityPanel fileId="file-123" />);

    expect(
      screen.getByText(/Computing Variance Inflation Factors and Condition Index.../i),
    ).toBeInTheDocument();
  });

  it("renders error alert with retry button when API fails", async () => {
    FileServices.getVIFMetrics.mockRejectedValueOnce(new Error("Network failure"));
    render(<VIFMulticollinearityPanel fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByText("Network failure")).toBeInTheDocument();
    });

    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();

    FileServices.getVIFMetrics.mockResolvedValueOnce(mockSuccessData);
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText("sqft_living")).toBeInTheDocument();
    });
  });

  it("renders empty state notice when fewer than 2 numeric features are present", async () => {
    FileServices.getVIFMetrics.mockResolvedValueOnce({
      vif_scores: [],
      constant_columns: ["zero_var_col"],
      summary: {
        overall_multicollinearity: "LOW",
        total_features_evaluated: 1,
        high_vif_features: 0,
      },
    });

    render(<VIFMulticollinearityPanel fileId="file-empty" />);

    await waitFor(() => {
      expect(
        screen.getByText(
          /At least 2 non-constant numeric features are required to compute Variance Inflation Factors/i,
        ),
      ).toBeInTheDocument();
      expect(screen.getByText(/zero_var_col/)).toBeInTheDocument();
    });
  });

  it("renders summary cards, severity badges, and feature table on success", async () => {
    FileServices.getVIFMetrics.mockResolvedValueOnce(mockSuccessData);
    render(<VIFMulticollinearityPanel fileId="file-success" />);

    await waitFor(() => {
      expect(screen.getByText("Multicollinearity Profiler")).toBeInTheDocument();
      expect(screen.getByText(/High Multicollinearity/i)).toBeInTheDocument();
    });

    // Check summary metric cards
    expect(screen.getByText("34.2")).toBeInTheDocument();
    expect(screen.getByText("Severe ill-conditioning")).toBeInTheDocument();

    // Check table rows
    expect(screen.getByTestId("vif-row-sqft_living")).toBeInTheDocument();
    expect(screen.getByText("12.45")).toBeInTheDocument();
    expect(screen.getByTestId("vif-row-sqft_above")).toBeInTheDocument();
    expect(screen.getByText("7.2")).toBeInTheDocument();
    expect(screen.getByTestId("vif-row-bedrooms")).toBeInTheDocument();
    expect(screen.getByText("2.15")).toBeInTheDocument();
    expect(screen.getByTestId("vif-row-sqft_duplicate")).toBeInTheDocument();
    expect(screen.getByText(/∞ \(Linear Dep\.\)/i)).toBeInTheDocument();

    // Notice for constant columns
    expect(screen.getByText(/dummy_constant/)).toBeInTheDocument();
  });

  it("toggles the VIF guide panel when guide button is clicked", async () => {
    FileServices.getVIFMetrics.mockResolvedValueOnce(mockSuccessData);
    render(<VIFMulticollinearityPanel fileId="file-guide" />);

    await waitFor(() => {
      expect(screen.getByTestId("toggle-vif-guide-btn")).toBeInTheDocument();
    });

    expect(screen.queryByTestId("vif-guide-panel")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("toggle-vif-guide-btn"));
    expect(screen.getByTestId("vif-guide-panel")).toBeInTheDocument();
    expect(
      screen.getByText(/Understanding Variance Inflation Factor \(VIF\)/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("toggle-vif-guide-btn"));
    expect(screen.queryByTestId("vif-guide-panel")).not.toBeInTheDocument();
  });

  it("filters table rows by severity", async () => {
    FileServices.getVIFMetrics.mockResolvedValueOnce(mockSuccessData);
    render(<VIFMulticollinearityPanel fileId="file-filter" />);

    await waitFor(() => {
      expect(screen.getByTestId("vif-row-sqft_living")).toBeInTheDocument();
    });

    // Filter HIGH
    fireEvent.click(screen.getByTestId("filter-vif-high"));
    expect(screen.getByTestId("vif-row-sqft_living")).toBeInTheDocument();
    expect(screen.getByTestId("vif-row-sqft_duplicate")).toBeInTheDocument();
    expect(screen.queryByTestId("vif-row-bedrooms")).not.toBeInTheDocument();
    expect(screen.queryByTestId("vif-row-sqft_above")).not.toBeInTheDocument();

    // Filter MODERATE
    fireEvent.click(screen.getByTestId("filter-vif-mod"));
    expect(screen.getByTestId("vif-row-sqft_above")).toBeInTheDocument();
    expect(screen.queryByTestId("vif-row-sqft_living")).not.toBeInTheDocument();

    // Filter LOW
    fireEvent.click(screen.getByTestId("filter-vif-low"));
    expect(screen.getByTestId("vif-row-bedrooms")).toBeInTheDocument();
    expect(screen.queryByTestId("vif-row-sqft_living")).not.toBeInTheDocument();

    // Filter ALL
    fireEvent.click(screen.getByTestId("filter-vif-all"));
    expect(screen.getByTestId("vif-row-sqft_living")).toBeInTheDocument();
    expect(screen.getByTestId("vif-row-bedrooms")).toBeInTheDocument();
  });

  it("re-fetches when refresh button is clicked", async () => {
    FileServices.getVIFMetrics.mockResolvedValue(mockSuccessData);
    render(<VIFMulticollinearityPanel fileId="file-refresh" />);

    await waitFor(() => {
      expect(FileServices.getVIFMetrics).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(screen.getByTestId("refresh-vif-btn"));
    await waitFor(() => {
      expect(FileServices.getVIFMetrics).toHaveBeenCalledTimes(2);
    });
  });
});
