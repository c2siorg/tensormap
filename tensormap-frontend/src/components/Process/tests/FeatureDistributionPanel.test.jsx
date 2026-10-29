import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import FeatureDistributionPanel from "../FeatureDistributionPanel";
import { getFeatureDistribution } from "../../../services/FileServices";

vi.mock("../../../services/FileServices", () => ({
  getFeatureDistribution: vi.fn(),
}));

const MOCK_DISTRIBUTION_DATA = {
  total_rows: 50,
  total_cols: 2,
  columns: [
    {
      column: "age",
      dtype: "int64",
      is_numeric: true,
      count: 50,
      null_count: 0,
      mean: 32.5,
      std: 8.2,
      min: 18.0,
      max: 95.0,
      q25: 26.0,
      median: 31.0,
      q75: 38.0,
      iqr: 12.0,
      skewness: 1.45,
      lower_bound: 8.0,
      upper_bound: 56.0,
      outlier_count: 2,
      outlier_percentage: 4.0,
      histogram: [
        { bin_start: 18.0, bin_end: 33.4, bin_label: "18 - 33.4", count: 30 },
        { bin_start: 33.4, bin_end: 48.8, bin_label: "33.4 - 48.8", count: 18 },
        { bin_start: 48.8, bin_end: 95.0, bin_label: "48.8 - 95", count: 2 },
      ],
    },
    {
      column: "department",
      dtype: "object",
      is_numeric: false,
      count: 50,
      null_count: 0,
      unique_count: 3,
      top_categories: [
        { category: "Engineering", count: 25, percentage: 50.0 },
        { category: "Design", count: 15, percentage: 30.0 },
        { category: "Marketing", count: 10, percentage: 20.0 },
      ],
    },
  ],
};

describe("FeatureDistributionPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state while fetching distribution data", () => {
    getFeatureDistribution.mockImplementation(() => new Promise(() => {}));
    render(<FeatureDistributionPanel fileId="file-123" />);

    expect(screen.getByTestId("distribution-loading")).toBeInTheDocument();
  });

  it("renders error message when fetch fails", async () => {
    getFeatureDistribution.mockRejectedValue(new Error("Network Error"));
    render(<FeatureDistributionPanel fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByTestId("distribution-error")).toBeInTheDocument();
      expect(screen.getByText("Failed to load feature distribution profiles")).toBeInTheDocument();
    });
  });

  it("renders empty state when no columns exist", async () => {
    getFeatureDistribution.mockResolvedValue({ total_rows: 0, total_cols: 0, columns: [] });
    render(<FeatureDistributionPanel fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByText("No distribution profiles available")).toBeInTheDocument();
    });
  });

  it("renders numeric distribution with outlier warning, metrics, and histogram", async () => {
    getFeatureDistribution.mockResolvedValue(MOCK_DISTRIBUTION_DATA);
    const { container } = render(<FeatureDistributionPanel fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Feature Distributions & Outlier Profiler/)).toBeInTheDocument();
    });

    // Check summary header badge
    expect(screen.getByText(/2 columns · 2 outliers detected/)).toBeInTheDocument();

    // Check outlier alert banner
    expect(screen.getByText(/2 Outliers Detected \(4% of column\)/i)).toBeInTheDocument();

    // Check statistics tiles
    expect(screen.getByText("32.5000")).toBeInTheDocument();
    expect(screen.getByText("31")).toBeInTheDocument();
    expect(screen.getByText(/IQR: 12/)).toBeInTheDocument();
    expect(screen.getByText(/Skewness:/)).toBeInTheDocument();
    expect(screen.getByText("1.4500")).toBeInTheDocument();

    // Check Recharts container
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("switches to categorical distribution view when selected from dropdown", async () => {
    const user = userEvent.setup();
    getFeatureDistribution.mockResolvedValue(MOCK_DISTRIBUTION_DATA);
    render(<FeatureDistributionPanel fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByLabelText("Select feature")).toBeInTheDocument();
    });

    const select = screen.getByLabelText("Select feature");
    await user.selectOptions(select, "department");

    // Check categorical tiles
    expect(screen.getByText("Unique Categories")).toBeInTheDocument();
    expect(screen.getByText("Total Samples")).toBeInTheDocument();

    // Check category table content
    expect(screen.getByText("Engineering")).toBeInTheDocument();
    expect(screen.getByText("25")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();

    expect(screen.getByText("Design")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
    expect(screen.getByText("30%")).toBeInTheDocument();
  });

  it("toggles collapsible content when header is clicked", async () => {
    const user = userEvent.setup();
    getFeatureDistribution.mockResolvedValue(MOCK_DISTRIBUTION_DATA);
    render(<FeatureDistributionPanel fileId="file-123" />);

    await waitFor(() => {
      expect(screen.getByLabelText("Select feature")).toBeInTheDocument();
    });

    const toggleBtn = screen.getByRole("button", {
      name: /Feature Distributions & Outlier Profiler/i,
    });
    expect(toggleBtn).toHaveAttribute("aria-expanded", "true");

    // Collapse
    await user.click(toggleBtn);
    expect(toggleBtn).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText("Select feature")).not.toBeInTheDocument();

    // Expand again
    await user.click(toggleBtn);
    expect(toggleBtn).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText("Select feature")).toBeInTheDocument();
  });
});
