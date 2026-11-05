/**
 * Tests for CalibrationCurvePanel component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import CalibrationCurvePanel from "../CalibrationCurvePanel";
import axios from "@/shared/Axios";

vi.mock("@/shared/Axios");

// Mock ResponsiveContainer to render children without 0x0 size errors in jsdom
vi.mock("recharts", async () => {
  const actual = await vi.importActual<any>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: any) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

describe("CalibrationCurvePanel", () => {
  const mockCalibrationData = {
    classes: [
      {
        class_index: 0,
        class_name: "Setosa",
        insufficient_data: false,
        brier_score: 0.0412,
        ece: 0.0354,
        mce: 0.0812,
        points: [
          {
            bin_index: 0,
            bin_label: "0.0 - 0.1",
            mean_predicted: 0.05,
            fraction_positives: 0.04,
            count: 15,
            error: 0.01,
          },
          {
            bin_index: 9,
            bin_label: "0.9 - 1.0",
            mean_predicted: 0.95,
            fraction_positives: 0.96,
            count: 20,
            error: 0.01,
          },
        ],
        histogram: [
          {
            bin_index: 0,
            bin_start: 0.0,
            bin_end: 0.1,
            bin_label: "0.0 - 0.1",
            count: 15,
            frequency: 0.4286,
          },
          {
            bin_index: 9,
            bin_start: 0.9,
            bin_end: 1.0,
            bin_label: "0.9 - 1.0",
            count: 20,
            frequency: 0.5714,
          },
        ],
      },
      {
        class_index: 1,
        class_name: "Versicolor",
        insufficient_data: false,
        brier_score: 0.0521,
        ece: 0.0412,
        mce: 0.0954,
        points: [
          {
            bin_index: 0,
            bin_label: "0.0 - 0.1",
            mean_predicted: 0.06,
            fraction_positives: 0.05,
            count: 10,
            error: 0.01,
          },
        ],
        histogram: [
          {
            bin_index: 0,
            bin_start: 0.0,
            bin_end: 0.1,
            bin_label: "0.0 - 0.1",
            count: 10,
            frequency: 1.0,
          },
        ],
      },
    ],
    macro_brier_score: 0.0467,
    macro_ece: 0.0383,
    macro_mce: 0.0883,
    is_binary: false,
    class_names: ["Setosa", "Versicolor"],
    n_samples: 35,
    n_bins: 10,
    analysis_type: "classification",
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders calibration metrics and header successfully", async () => {
    (axios.get as any).mockResolvedValue({ data: mockCalibrationData });

    render(<CalibrationCurvePanel jobId="test-job-calib" />);

    await waitFor(() => {
      expect(screen.getByText("Probability Calibration & Reliability Curve")).toBeInTheDocument();
      expect(screen.getByText("Brier Score: 0.0412")).toBeInTheDocument();
      expect(screen.getByText("ECE: 3.54%")).toBeInTheDocument();
      expect(screen.getByText("MCE: 8.12%")).toBeInTheDocument();
      expect(screen.getByText("Macro ECE: 3.83%")).toBeInTheDocument();
    });
  });

  it("renders error state when API call fails", async () => {
    (axios.get as any).mockRejectedValue(new Error("Network Error"));

    render(<CalibrationCurvePanel jobId="test-job-calib" />);

    await waitFor(() => {
      expect(screen.getByText("Failed to load calibration data")).toBeInTheDocument();
      expect(screen.getByText("Network Error")).toBeInTheDocument();
    });
  });

  it("renders insufficient data fallback when class has insufficient data", async () => {
    const insufficientData = {
      ...mockCalibrationData,
      classes: [
        {
          class_index: 0,
          class_name: "RareClass",
          insufficient_data: true,
          brier_score: null,
          ece: null,
          mce: null,
          points: [],
          histogram: [],
        },
      ],
      class_names: ["RareClass"],
    };

    (axios.get as any).mockResolvedValue({ data: insufficientData });

    render(<CalibrationCurvePanel jobId="test-job-calib" />);

    await waitFor(() => {
      expect(screen.getByText("Insufficient Class Samples")).toBeInTheDocument();
      expect(
        screen.getByText(
          /The test dataset does not contain sufficient positive and negative examples/i,
        ),
      ).toBeInTheDocument();
    });
  });

  it("toggles bin details breakdown table", async () => {
    (axios.get as any).mockResolvedValue({ data: mockCalibrationData });

    render(<CalibrationCurvePanel jobId="test-job-calib" />);

    await waitFor(() => {
      expect(screen.getByText("Bin Details & Calibration Error Breakdown")).toBeInTheDocument();
    });

    const toggleButton = screen.getByText("Bin Details & Calibration Error Breakdown");
    fireEvent.click(toggleButton);

    await waitFor(() => {
      expect(screen.getByText("Bin Interval")).toBeInTheDocument();
      expect(screen.getByText("Mean Confidence")).toBeInTheDocument();
      expect(screen.getByText("Empirical Accuracy")).toBeInTheDocument();
    });
  });
});
