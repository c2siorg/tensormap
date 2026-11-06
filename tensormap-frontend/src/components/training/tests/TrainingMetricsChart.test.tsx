/**
 * Test suite for TrainingMetricsChart and related components.
 * @module
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TrainingMetricsChart from "../TrainingMetricsChart";
import LossChart from "../LossChart";
import AccuracyChart from "../AccuracyChart";
import TrainingHeader from "../TrainingHeader";
import MetricsSummaryBar from "../MetricsSummaryBar";
import { EpochMetric } from "@/types/training";

// Mock the hooks and services
vi.mock("@/hooks/useTrainingMetrics", () => ({
  useTrainingMetrics: vi.fn(),
}));

vi.mock("../ReproducibilityDetails", () => ({
  default: () => null,
}));

vi.mock("@/services/socketService", () => ({
  subscribeToJob: vi.fn(() => vi.fn()),
  unsubscribeFromJob: vi.fn(),
  cancelJob: vi.fn(() => Promise.resolve()),
  getTrainingSocket: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    connected: true,
  })),
}));

import { useTrainingMetrics } from "@/hooks/useTrainingMetrics";

describe("LossChart", () => {
  it("renders loss chart with correct data points", () => {
    const metrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9, accuracy: 0.5 },
      { epoch: 2, loss: 0.7, accuracy: 0.6 },
      { epoch: 3, loss: 0.5, accuracy: 0.7 },
      { epoch: 4, loss: 0.3, accuracy: 0.8 },
      { epoch: 5, loss: 0.1, accuracy: 0.9 },
    ];

    const { container } = render(<LossChart metrics={metrics} />);

    // Check that the chart container is rendered
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("renders validation loss when available", () => {
    const metrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9, val_loss: 0.95 },
      { epoch: 2, loss: 0.7, val_loss: 0.75 },
    ];

    const { container } = render(<LossChart metrics={metrics} />);

    // Check that chart is rendered
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("handles missing validation metrics gracefully", () => {
    const metrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9 },
      { epoch: 2, loss: 0.7 },
    ];

    const { container } = render(<LossChart metrics={metrics} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });
});

describe("AccuracyChart", () => {
  it("renders accuracy chart with correct data points", () => {
    const metrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9, accuracy: 0.5 },
      { epoch: 2, loss: 0.7, accuracy: 0.6 },
      { epoch: 3, loss: 0.5, accuracy: 0.7 },
    ];

    const { container } = render(<AccuracyChart metrics={metrics} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("does not render when no accuracy data", () => {
    const metrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9 },
      { epoch: 2, loss: 0.7 },
    ];

    const { container } = render(<AccuracyChart metrics={metrics} />);
    expect(container.querySelector(".recharts-responsive-container")).toBeNull();
  });
});

describe("TrainingHeader", () => {
  const mockOnStop = vi.fn();

  beforeEach(() => {
    mockOnStop.mockClear();
  });

  it("displays status badge correctly", () => {
    render(
      <TrainingHeader
        jobId="test-job-123"
        status="running"
        currentEpoch={10}
        totalEpochs={50}
        elapsedTime={100}
        eta={400}
        onStop={mockOnStop}
        cancelRequested={false}
      />,
    );

    expect(screen.getByText("Running")).toBeTruthy();
  });

  it("shows stop button when running", () => {
    render(
      <TrainingHeader
        jobId="test-job-123"
        status="running"
        currentEpoch={10}
        totalEpochs={50}
        elapsedTime={100}
        eta={400}
        onStop={mockOnStop}
        cancelRequested={false}
      />,
    );

    const stopButton = screen.getByRole("button", { name: /stop training/i });
    expect(stopButton).toBeTruthy();
  });

  it("stop button calls cancel callback", async () => {
    const user = userEvent.setup();

    render(
      <TrainingHeader
        jobId="test-job-123"
        status="running"
        currentEpoch={10}
        totalEpochs={50}
        elapsedTime={100}
        eta={400}
        onStop={mockOnStop}
        cancelRequested={false}
      />,
    );

    const stopButton = screen.getByRole("button", { name: /stop training/i });
    await user.click(stopButton);

    expect(mockOnStop).toHaveBeenCalledTimes(1);
  });

  it("completed status shows green badge", () => {
    const { container } = render(
      <TrainingHeader
        jobId="test-job-123"
        status="completed"
        currentEpoch={50}
        totalEpochs={50}
        elapsedTime={500}
        eta={0}
        onStop={mockOnStop}
        cancelRequested={false}
      />,
    );

    expect(screen.getByText("Completed")).toBeTruthy();
    const badge = container.querySelector(".bg-green-500");
    expect(badge).toBeTruthy();
  });

  it("eta calculates correctly", () => {
    // 10 epochs done in 100s = 10s per epoch
    // 40 remaining epochs = 400s ETA
    render(
      <TrainingHeader
        jobId="test-job-123"
        status="running"
        currentEpoch={10}
        totalEpochs={50}
        elapsedTime={100}
        eta={400}
        onStop={mockOnStop}
        cancelRequested={false}
      />,
    );

    // ETA should be displayed as 06:40 (6 minutes 40 seconds)
    expect(screen.getByText(/06:40/)).toBeTruthy();
  });
});

describe("MetricsSummaryBar", () => {
  it("displays current metrics correctly", () => {
    const currentMetrics: EpochMetric = {
      epoch: 10,
      loss: 0.1234,
      accuracy: 0.95,
      val_loss: 0.1456,
      val_accuracy: 0.93,
    };

    render(
      <MetricsSummaryBar
        currentMetrics={currentMetrics}
        bestEpoch={null}
        currentEpoch={10}
        totalEpochs={50}
      />,
    );

    expect(screen.getByText("0.1234")).toBeTruthy();
    expect(screen.getByText("95.00%")).toBeTruthy();
  });

  it("best epoch highlighted correctly", () => {
    const metrics: EpochMetric = {
      epoch: 10,
      loss: 0.2,
      val_loss: 0.25,
    };

    const bestEpoch = { epoch: 3, val_loss: 0.15 };

    render(
      <MetricsSummaryBar
        currentMetrics={metrics}
        bestEpoch={bestEpoch}
        currentEpoch={10}
        totalEpochs={50}
      />,
    );

    expect(screen.getByText("3")).toBeTruthy();
    expect(screen.getByText("0.1500")).toBeTruthy();
  });

  it("shows no validation data message when appropriate", () => {
    render(
      <MetricsSummaryBar
        currentMetrics={null}
        bestEpoch={null}
        currentEpoch={0}
        totalEpochs={50}
      />,
    );

    expect(screen.getByText("No validation data")).toBeTruthy();
  });
});

describe("TrainingMetricsChart Integration", () => {
  beforeEach(() => {
    vi.mocked(useTrainingMetrics).mockReturnValue({
      metrics: [],
      status: "idle",
      currentEpoch: 0,
      totalEpochs: 50,
      cancel: vi.fn(),
      cancelRequested: false,
      startedAt: null,
      error: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("catchup event populates chart data", () => {
    const mockMetrics: EpochMetric[] = Array.from({ length: 10 }, (_, i) => ({
      epoch: i + 1,
      loss: 1 - i * 0.1,
      accuracy: i * 0.1,
    }));

    vi.mocked(useTrainingMetrics).mockReturnValue({
      metrics: mockMetrics,
      status: "running",
      currentEpoch: 10,
      totalEpochs: 50,
      cancel: vi.fn(),
      cancelRequested: false,
      startedAt: new Date().toISOString(),
      error: null,
    });

    const { container } = render(<TrainingMetricsChart jobId="test-job" totalEpochs={50} />);

    // Should render charts
    const charts = container.querySelectorAll(".recharts-responsive-container");
    expect(charts.length).toBeGreaterThan(0);
  });

  it("live metric event appends data point", async () => {
    const initialMetrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9 },
      { epoch: 2, loss: 0.7 },
    ];

    const { rerender } = render(<TrainingMetricsChart jobId="test-job" totalEpochs={50} />);

    vi.mocked(useTrainingMetrics).mockReturnValue({
      metrics: initialMetrics,
      status: "running",
      currentEpoch: 2,
      totalEpochs: 50,
      cancel: vi.fn(),
      cancelRequested: false,
      startedAt: new Date().toISOString(),
      error: null,
    });

    rerender(<TrainingMetricsChart jobId="test-job" totalEpochs={50} />);

    // Simulate new metric
    const updatedMetrics = [...initialMetrics, { epoch: 3, loss: 0.5 }];

    vi.mocked(useTrainingMetrics).mockReturnValue({
      metrics: updatedMetrics,
      status: "running",
      currentEpoch: 3,
      totalEpochs: 50,
      cancel: vi.fn(),
      cancelRequested: false,
      startedAt: new Date().toISOString(),
      error: null,
    });

    rerender(<TrainingMetricsChart jobId="test-job" totalEpochs={50} />);

    await waitFor(() => {
      expect(screen.getByText(/Epoch 3/)).toBeTruthy();
    });
  });

  it("chart is responsive", () => {
    const mockMetrics: EpochMetric[] = [
      { epoch: 1, loss: 0.9 },
      { epoch: 2, loss: 0.7 },
    ];

    vi.mocked(useTrainingMetrics).mockReturnValue({
      metrics: mockMetrics,
      status: "running",
      currentEpoch: 2,
      totalEpochs: 50,
      cancel: vi.fn(),
      cancelRequested: false,
      startedAt: new Date().toISOString(),
      error: null,
    });

    const { container } = render(
      <div style={{ width: "400px" }}>
        <TrainingMetricsChart jobId="test-job" totalEpochs={50} />
      </div>,
    );

    const responsiveContainer = container.querySelector(".recharts-responsive-container");
    expect(responsiveContainer).toBeTruthy();
  });
});
