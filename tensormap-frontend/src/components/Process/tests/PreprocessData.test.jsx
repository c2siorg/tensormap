import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PreprocessData from "../PreprocessData";
import { getCovMatrix } from "../../../services/FileServices";

vi.mock("../../../services/FileServices", () => ({
  getCovMatrix: vi.fn(),
  transformData: vi.fn(),
}));

vi.mock("../TransformationCreator", () => ({
  default: ({ onAdd }) => (
    <button
      data-testid="mock-add-transformation"
      onClick={() => onAdd({ feature: "colA", transformation: "One Hot Encoding" })}
    >
      Mock Add Transformation
    </button>
  ),
}));

describe("PreprocessData Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not call getCovMatrix when fileId is null or empty", () => {
    render(<PreprocessData fileId={null} updateFileList={vi.fn()} />);
    expect(getCovMatrix).not.toHaveBeenCalled();
  });

  it("fetches column metadata when fileId is provided", async () => {
    getCovMatrix.mockResolvedValueOnce({
      data_types: { age: "int64", income: "float64" },
    });

    render(<PreprocessData fileId="file-1" updateFileList={vi.fn()} />);

    expect(getCovMatrix).toHaveBeenCalledWith("file-1");
    await waitFor(() => {
      expect(screen.getByTestId("mock-add-transformation")).toBeTruthy();
    });
  });

  it("displays error message when getCovMatrix fails", async () => {
    getCovMatrix.mockRejectedValueOnce(new Error("Network error"));

    render(<PreprocessData fileId="file-1" updateFileList={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText("Network error")).toBeTruthy();
    });
  });

  it("resets transformations when fileId prop changes", async () => {
    const user = userEvent.setup();
    getCovMatrix.mockResolvedValue({
      data_types: { colA: "int64", colB: "float64" },
    });

    const { rerender } = render(<PreprocessData fileId="file-1" updateFileList={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByTestId("mock-add-transformation")).toBeTruthy();
    });

    // Add a transformation
    await user.click(screen.getByTestId("mock-add-transformation"));

    // Apply Transformations button should become enabled because a transformation was added
    const applyBtn = screen.getByRole("button", { name: /apply transformations/i });
    expect(applyBtn).toBeEnabled();

    // Change fileId to file-2
    rerender(<PreprocessData fileId="file-2" updateFileList={vi.fn()} />);

    // Transformations should be cleared, so Apply Transformations button is disabled again
    await waitFor(() => {
      expect(applyBtn).toBeDisabled();
    });
    expect(getCovMatrix).toHaveBeenCalledWith("file-2");
  });
});
