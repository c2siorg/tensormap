/**
 * Test suite for CopyableText component.
 * @module
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CopyableText from "../CopyableText";
import logger from "@/shared/logger";

// Mock logger
vi.mock("@/shared/logger", () => ({
  default: {
    error: vi.fn(),
  },
}));

describe("CopyableText", () => {
  beforeEach(() => {
    vi.mocked(logger.error).mockClear();
  });

  it("renders text value", () => {
    render(<CopyableText value="test-value" />);
    expect(screen.getByText("test-value")).toBeTruthy();
  });

  it("renders with custom label", () => {
    render(<CopyableText value="test-value" label="custom label" />);
    const button = screen.getByRole("button", { name: /copy custom label/i });
    expect(button).toBeTruthy();
  });

  it("copies value to clipboard on button click", async () => {
    const user = userEvent.setup();
    render(<CopyableText value="test-fingerprint-123" />);

    // Spy AFTER render to avoid Happy-DOM resetting it
    const writeTextSpy = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();

    const copyButton = screen.getByRole("button", { name: /copy/i });
    await user.click(copyButton);

    expect(writeTextSpy).toHaveBeenCalledWith("test-fingerprint-123");
    writeTextSpy.mockRestore();
  });

  it("shows check icon after successful copy", async () => {
    const user = userEvent.setup();
    const { container } = render(<CopyableText value="test-value" />);

    const copyButton = screen.getByRole("button", { name: /copy/i });
    await user.click(copyButton);

    // Check icon should be visible (green check mark)
    const checkIcon = container.querySelector(".text-green-600");
    expect(checkIcon).toBeTruthy();
  });

  it("applies truncate class by default", () => {
    const { container } = render(<CopyableText value="test-value" />);
    const textElement = container.querySelector(".truncate");
    expect(textElement).toBeTruthy();
  });

  it("applies break-all class when truncate is false", () => {
    const { container } = render(<CopyableText value="test-value" truncate={false} />);
    const textElement = container.querySelector(".break-all");
    expect(textElement).toBeTruthy();
  });

  it("applies custom className", () => {
    const { container } = render(<CopyableText value="test" className="custom-class" />);
    expect(container.querySelector(".custom-class")).toBeTruthy();
  });

  it("sets title attribute with full value", () => {
    const longValue = "a".repeat(100);
    const { container } = render(<CopyableText value={longValue} />);
    const textElement = container.querySelector(`[title="${longValue}"]`);
    expect(textElement).toBeTruthy();
  });

  it("handles clipboard write failure without showing success feedback", async () => {
    const user = userEvent.setup();
    const { container } = render(<CopyableText value="test-value" />);

    // Spy AFTER render to avoid Happy-DOM resetting it
    const clipboardError = new Error("Permission denied");
    const writeTextSpy = vi.spyOn(navigator.clipboard, "writeText").mockRejectedValueOnce(clipboardError);

    const copyButton = screen.getByRole("button", { name: /copy/i });
    await user.click(copyButton);

    // Logger should be called with the error
    expect(logger.error).toHaveBeenCalledWith(
      "Failed to copy to clipboard:",
      clipboardError,
    );

    // Check icon should NOT be visible (no success feedback)
    const checkIcon = container.querySelector(".text-green-600");
    expect(checkIcon).toBeNull();
    writeTextSpy.mockRestore();
  });
});
