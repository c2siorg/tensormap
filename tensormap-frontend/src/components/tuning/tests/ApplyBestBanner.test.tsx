/**
 * Test suite for ApplyBestBanner component.
 * @module
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ApplyBestBanner from "../ApplyBestBanner";

describe("ApplyBestBanner", () => {
  const mockBestParams = {
    optimizer: "adam",
    learning_rate: 0.0011,
    batch_size: 16,
    epochs: 50,
  };

  const mockOnApply = vi.fn();

  beforeEach(() => {
    mockOnApply.mockClear();
  });

  it("shows correct params", () => {
    render(
      <ApplyBestBanner
        bestParams={mockBestParams}
        metric="val_accuracy"
        metricValue={0.841}
        onApply={mockOnApply}
      />,
    );

    expect(screen.getByText("adam")).toBeTruthy();
    expect(screen.getByText("1.10e-3")).toBeTruthy();
    expect(screen.getByText("16")).toBeTruthy();
    expect(screen.getByText("50")).toBeTruthy();
    expect(screen.getByText("val_accuracy: 0.8410")).toBeTruthy();
  });

  it("apply-best button calls POST endpoint", async () => {
    const user = userEvent.setup();

    mockOnApply.mockResolvedValue(undefined);

    render(
      <ApplyBestBanner
        bestParams={mockBestParams}
        metric="val_accuracy"
        metricValue={0.841}
        onApply={mockOnApply}
      />,
    );

    const applyButton = screen.getByRole("button", { name: /Apply Best Parameters/i });
    await user.click(applyButton);

    await waitFor(() => {
      expect(mockOnApply).toHaveBeenCalledTimes(1);
    });
  });

  it("shows loading state while applying", async () => {
    const user = userEvent.setup();

    mockOnApply.mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(resolve, 100);
        }),
    );

    render(
      <ApplyBestBanner
        bestParams={mockBestParams}
        metric="val_accuracy"
        metricValue={0.841}
        onApply={mockOnApply}
      />,
    );

    const applyButton = screen.getByRole("button", { name: /Apply Best Parameters/i });
    await user.click(applyButton);

    // Check for loading state
    expect(screen.getByText(/Applying/i)).toBeTruthy();
    expect(applyButton).toBeDisabled();

    await waitFor(() => {
      expect(mockOnApply).toHaveBeenCalled();
    });
  });

  it("shows success state after applying", async () => {
    const user = userEvent.setup();

    mockOnApply.mockResolvedValue(undefined);

    render(
      <ApplyBestBanner
        bestParams={mockBestParams}
        metric="val_accuracy"
        metricValue={0.841}
        onApply={mockOnApply}
      />,
    );

    const applyButton = screen.getByRole("button", { name: /Apply Best Parameters/i });
    await user.click(applyButton);

    await waitFor(() => {
      expect(screen.getByText(/✓ Best parameters applied/i)).toBeTruthy();
      // Use getAllByText since "Applied" appears twice (in heading and badge)
      const appliedTexts = screen.getAllByText(/Applied/i);
      expect(appliedTexts.length).toBeGreaterThan(0);
    });
  });

  it("shows error message on failure", async () => {
    const user = userEvent.setup();

    mockOnApply.mockRejectedValue(new Error("Network error"));

    render(
      <ApplyBestBanner
        bestParams={mockBestParams}
        metric="val_accuracy"
        metricValue={0.841}
        onApply={mockOnApply}
      />,
    );

    const applyButton = screen.getByRole("button", { name: /Apply Best Parameters/i });
    await user.click(applyButton);

    await waitFor(() => {
      expect(screen.getByText(/Network error/i)).toBeTruthy();
    });
  });

  it("displays all hyperparameter fields", () => {
    render(
      <ApplyBestBanner
        bestParams={mockBestParams}
        metric="val_accuracy"
        metricValue={0.841}
        onApply={mockOnApply}
      />,
    );

    expect(screen.getByText(/OPTIMIZER/i)).toBeTruthy();
    expect(screen.getByText(/LEARNING RATE/i)).toBeTruthy();
    expect(screen.getByText(/BATCH SIZE/i)).toBeTruthy();
    expect(screen.getByText(/EPOCHS/i)).toBeTruthy();
  });

  it("handles partial params gracefully", () => {
    const partialParams = {
      optimizer: "sgd",
      learning_rate: 0.01,
    };

    render(
      <ApplyBestBanner
        bestParams={partialParams}
        metric="val_loss"
        metricValue={0.123}
        onApply={mockOnApply}
      />,
    );

    expect(screen.getByText("sgd")).toBeTruthy();
    expect(screen.getByText("1.00e-2")).toBeTruthy();
    expect(screen.getByText("val_loss: 0.1230")).toBeTruthy();
  });
});
