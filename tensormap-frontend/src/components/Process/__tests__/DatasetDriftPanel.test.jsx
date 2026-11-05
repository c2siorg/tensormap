import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import DatasetDriftPanel from "../DatasetDriftPanel";
import { getDatasetDrift } from "../../../services/FileServices";

vi.mock("../../../services/FileServices", () => ({
  getDatasetDrift: vi.fn(),
}));

vi.mock("recharts", async () => {
  const actual = await vi.importActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  };
});

describe("DatasetDriftPanel Component", () => {
  const mockReport = {
    dataset_drift_detected: true,
    drift_share: 0.5,
    drifted_features_count: 1,
    total_features_count: 2,
    mode: "train_test_split",
    alpha: 0.05,
    baseline: {
      name: "Training Split (80%)",
      rows: 80,
    },
    target: {
      name: "Testing Split (20%)",
      rows: 20,
    },
    features: [
      {
        feature_name: "feature_drifted",
        data_type: "numeric",
        drift_detected: true,
        drift_severity: "severe",
        insufficient_data: false,
        ks_statistic: 0.45,
        ks_pvalue: 0.002,
        wasserstein_distance: 3.2,
        normalized_wasserstein: 1.5,
        psi: 0.32,
        stats: {
          baseline: { count: 80, mean: 1.2, std: 0.5, min: 0.1, max: 2.5, missing_pct: 0 },
          target: { count: 20, mean: 4.5, std: 0.8, min: 2.8, max: 6.0, missing_pct: 0 },
        },
        distribution: [
          {
            bin_label: "[0.10, 2.00]",
            baseline_count: 60,
            target_count: 2,
            baseline_pct: 0.75,
            target_pct: 0.1,
          },
          {
            bin_label: "(2.00, 6.00]",
            baseline_count: 20,
            target_count: 18,
            baseline_pct: 0.25,
            target_pct: 0.9,
          },
        ],
      },
      {
        feature_name: "feature_stable",
        data_type: "numeric",
        drift_detected: false,
        drift_severity: "none",
        insufficient_data: false,
        ks_statistic: 0.12,
        ks_pvalue: 0.65,
        wasserstein_distance: 0.2,
        normalized_wasserstein: 0.1,
        psi: 0.04,
        stats: {
          baseline: { count: 80, mean: 10.0, std: 2.0, min: 5.0, max: 15.0, missing_pct: 0 },
          target: { count: 20, mean: 10.2, std: 2.1, min: 5.5, max: 14.8, missing_pct: 0 },
        },
        distribution: [],
      },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders drift header, alert banner, and metric badges", async () => {
    getDatasetDrift.mockResolvedValue(mockReport);

    render(<DatasetDriftPanel fileId="file-1" files={[]} />);

    await waitFor(() => {
      expect(screen.getByText(/Dataset Drift & Covariate Shift Profiler/i)).toBeInTheDocument();
      expect(screen.getByText("Dataset Drift Detected")).toBeInTheDocument();
      expect(screen.getByText(/1 of 2 features drifted/i)).toBeInTheDocument();
      expect(screen.getAllByText("feature_drifted").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("feature_stable")).toBeInTheDocument();
      expect(screen.getByText("Severe Drift")).toBeInTheDocument();
      expect(screen.getByText("Stable")).toBeInTheDocument();
    });
  });

  it("renders stable banner when no drift is detected", async () => {
    const stableReport = {
      ...mockReport,
      dataset_drift_detected: false,
      drift_share: 0.0,
      drifted_features_count: 0,
      features: [
        {
          ...mockReport.features[1],
        },
      ],
    };
    getDatasetDrift.mockResolvedValue(stableReport);

    render(<DatasetDriftPanel fileId="file-1" files={[]} />);

    await waitFor(() => {
      expect(screen.getByText("Distributions are Stable")).toBeInTheDocument();
    });
  });

  it("handles error state gracefully", async () => {
    getDatasetDrift.mockRejectedValue(new Error("Network Error"));

    render(<DatasetDriftPanel fileId="file-1" files={[]} />);

    await waitFor(() => {
      expect(screen.getByText("Could not profile dataset drift")).toBeInTheDocument();
      expect(screen.getByText("Network Error")).toBeInTheDocument();
    });
  });

  it("switches modes between split and compare datasets", async () => {
    getDatasetDrift.mockResolvedValue(mockReport);

    const otherFiles = [
      { file_id: "file-1", file_name: "current.csv", file_type: "csv" },
      { file_id: "file-2", file_name: "baseline.csv", file_type: "csv" },
    ];

    render(<DatasetDriftPanel fileId="file-1" files={otherFiles} />);

    await waitFor(() => {
      expect(screen.getByText("Train/Test Split")).toBeInTheDocument();
      expect(screen.getByText("Compare Datasets")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Compare Datasets"));

    await waitFor(() => {
      expect(screen.getByText("Select baseline dataset...")).toBeInTheDocument();
    });
  });
});
