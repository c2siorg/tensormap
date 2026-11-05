/**
 * Tests for LayerHealthPanel component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import LayerHealthPanel from "../LayerHealthPanel";
import axios from "@/shared/Axios";

vi.mock("@/shared/Axios");

describe("LayerHealthPanel", () => {
  const mockDiagnosticsData = {
    job_id: "test-job-456",
    overall_status: "HEALTHY",
    total_layers: 2,
    healthy_layers: 2,
    total_parameters: 1650,
    layers: [
      {
        layer_name: "dense_1",
        layer_type: "Dense",
        shape: [10, 64],
        param_count: 640,
        l2_norm: 1.2543,
        mean: 0.012,
        std: 0.154,
        min: -0.45,
        max: 0.52,
        dead_weights_ratio: 0.01,
        status: "HEALTHY",
        histogram: [
          { bin_start: -0.5, bin_end: 0.0, count: 320 },
          { bin_start: 0.0, bin_end: 0.5, count: 320 },
        ],
      },
      {
        layer_name: "dense_output",
        layer_type: "Dense",
        shape: [64, 2],
        param_count: 128,
        l2_norm: 0.892,
        mean: -0.005,
        std: 0.201,
        min: -0.6,
        max: 0.58,
        dead_weights_ratio: 0.0,
        status: "HEALTHY",
        histogram: [],
      },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state initially", () => {
    (axios.get as any).mockReturnValue(new Promise(() => {}));

    render(<LayerHealthPanel jobId="test-job-456" />);

    expect(screen.getByText(/Profiling layer weights and model health/i)).toBeInTheDocument();
  });

  it("renders error alert when API call fails", async () => {
    (axios.get as any).mockRejectedValueOnce({
      response: { data: { detail: "Model weights not found" } },
    });

    render(<LayerHealthPanel jobId="test-job-456" />);

    await waitFor(() => {
      expect(screen.getByText(/Layer Diagnostics Error:/i)).toBeInTheDocument();
      expect(screen.getByText(/Model weights not found/i)).toBeInTheDocument();
    });
  });

  it("renders layer statistics and healthy badges correctly on success", async () => {
    (axios.get as any).mockResolvedValueOnce({
      data: mockDiagnosticsData,
    });

    render(<LayerHealthPanel jobId="test-job-456" />);

    await waitFor(() => {
      expect(
        screen.getByText("Neural Network Layer Health & Weight Diagnostics")
      ).toBeInTheDocument();
      expect(screen.getByText("1,650")).toBeInTheDocument();
      expect(screen.getByText("2 / 2")).toBeInTheDocument();
      expect(screen.getByText("dense_1")).toBeInTheDocument();
      expect(screen.getByText("dense_output")).toBeInTheDocument();
      expect(screen.getByText("1.2543")).toBeInTheDocument();
    });
  });

  it("renders warning status badge when status is VANISHING_WEIGHTS", async () => {
    const warningData = {
      ...mockDiagnosticsData,
      overall_status: "WARNING",
      healthy_layers: 1,
      layers: [
        {
          ...mockDiagnosticsData.layers[0],
          status: "VANISHING_WEIGHTS",
        },
      ],
    };

    (axios.get as any).mockResolvedValueOnce({
      data: warningData,
    });

    render(<LayerHealthPanel jobId="test-job-456" />);

    await waitFor(() => {
      expect(screen.getByText("Vanishing")).toBeInTheDocument();
      expect(screen.getByText("Warning")).toBeInTheDocument();
    });
  });
});
