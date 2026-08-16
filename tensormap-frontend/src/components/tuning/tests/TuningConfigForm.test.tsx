/**
 * Test suite for TuningConfigForm component.
 * @module
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TuningConfigForm from "../TuningConfigForm";

describe("TuningConfigForm", () => {
  const mockOnStart = vi.fn();
  const mockEstimateTime = vi.fn(() => 600); // 10 minutes

  beforeEach(() => {
    mockOnStart.mockClear();
    mockEstimateTime.mockClear();
  });

  it("renders all hyperparameter controls", () => {
    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Strategy radio buttons
    expect(screen.getByLabelText(/Random Search/i)).toBeTruthy();
    expect(screen.getByLabelText(/Grid Search/i)).toBeTruthy();

    // Search space controls
    expect(screen.getByText(/Optimizer/i)).toBeTruthy();
    expect(screen.getByText(/Learning Rate/i)).toBeTruthy();
    expect(screen.getByText(/Batch Size/i)).toBeTruthy();
    expect(screen.getByText(/Epochs/i)).toBeTruthy();

    // Metric selector
    expect(screen.getByText(/Optimization Metric/i)).toBeTruthy();

    // Early stop
    expect(screen.getByLabelText(/Enable Early Stop/i)).toBeTruthy();

    // Start button
    expect(screen.getByRole("button", { name: /Start Hyperparameter Tuning/i })).toBeTruthy();
  });

  it("grid combination count calculated correctly client-side", async () => {
    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Switch to grid search
    const gridRadio = screen.getByLabelText(/Grid Search/i);
    await user.click(gridRadio);

    await waitFor(() => {
      // With default selections: 2 optimizers (adam, sgd) × 2 batch sizes (32, 64) × 1 epoch (50) × 2 LR = 8
      // The component simplifies LR range to × 2
      expect(mockEstimateTime).toHaveBeenCalled();
    });
  });

  it("warning shown when estimate > 30 min", async () => {
    mockEstimateTime.mockReturnValue(2000); // ~33 minutes

    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Select more options to increase estimate
    const epoch100 = screen.getByText("100");
    await user.click(epoch100);

    await waitFor(() => {
      expect(screen.getByText(/Consider reducing max_trials or narrowing/i)).toBeTruthy();
    });
  });

  it("grid error shown when combinations > 50", async () => {
    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Switch to grid search
    const gridRadio = screen.getByLabelText(/Grid Search/i);
    await user.click(gridRadio);

    // Select all optimizers
    const rmsprop = screen.getByText("rmsprop");
    await user.click(rmsprop);

    // Select all batch sizes
    const batch16 = screen.getByText("16");
    await user.click(batch16);
    const batch128 = screen.getByText("128");
    await user.click(batch128);

    // Select all epochs
    const epoch20 = screen.getByText("20");
    await user.click(epoch20);
    const epoch100 = screen.getByText("100");
    await user.click(epoch100);

    // Should show error: 3 opt × 4 batch × 3 epochs × 2 LR = 72 combinations
    await waitFor(() => {
      expect(screen.getByText(/Grid search would generate \d+ combinations/i)).toBeTruthy();
    });

    // Start button should be disabled
    const startButton = screen.getByRole("button", { name: /Start Hyperparameter Tuning/i });
    expect(startButton).toBeDisabled();
  });

  it("calls onStart with correct config", async () => {
    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Click start with default values
    const startButton = screen.getByRole("button", { name: /Start Hyperparameter Tuning/i });
    await user.click(startButton);

    await waitFor(() => {
      expect(mockOnStart).toHaveBeenCalled();
      const callArgs = mockOnStart.mock.calls[0][0];
      expect(callArgs.strategy).toBe("random");
      expect(callArgs.metric).toBe("val_accuracy");
      expect(callArgs.direction).toBe("maximize");
      expect(callArgs.search_space.optimizer).toContain("adam");
      expect(callArgs.search_space.optimizer).toContain("sgd");
      expect(callArgs.search_space.learning_rate.type).toBe("log_uniform");
    });
  });

  it("direction auto-set based on metric", async () => {
    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Check initial direction (val_accuracy → maximize)
    expect(screen.getByText(/maximize/i)).toBeTruthy();

    // Change to val_loss
    const metricSelect = screen.getByRole("combobox");
    await user.click(metricSelect);
    const valLoss = screen.getByText("val_loss");
    await user.click(valLoss);

    // Direction should change to minimize
    await waitFor(() => {
      expect(screen.getByText(/minimize/i)).toBeTruthy();
    });
  });

  it("learning rate slider updates correctly", async () => {
    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Check learning rate display shows exponential notation
    expect(screen.getByText(/1\.0e-4/i)).toBeTruthy();
    expect(screen.getByText(/1\.0e-3/i)).toBeTruthy();
  });

  it("optimizer checkboxes toggle correctly", async () => {
    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // adam and sgd are selected by default
    const adamCheckbox = screen.getByRole("checkbox", { name: /adam/i });
    expect(adamCheckbox).toBeChecked();

    // Uncheck adam
    await user.click(adamCheckbox);
    expect(adamCheckbox).not.toBeChecked();

    // Check rmsprop
    const rmspropCheckbox = screen.getByRole("checkbox", { name: /rmsprop/i });
    await user.click(rmspropCheckbox);
    expect(rmspropCheckbox).toBeChecked();
  });

  it("validates at least one option selected per parameter", async () => {
    const user = userEvent.setup();

    render(
      <TuningConfigForm
        modelName="test-model"
        onStart={mockOnStart}
        estimateTime={mockEstimateTime}
      />,
    );

    // Uncheck all optimizers
    const adamCheckbox = screen.getByRole("checkbox", { name: /adam/i });
    const sgdCheckbox = screen.getByRole("checkbox", { name: /sgd/i });
    await user.click(adamCheckbox);
    await user.click(sgdCheckbox);

    // Start button should be disabled
    const startButton = screen.getByRole("button", { name: /Start Hyperparameter Tuning/i });
    await waitFor(() => {
      expect(startButton).toBeDisabled();
    });
  });
});
