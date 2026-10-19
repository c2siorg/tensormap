import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getDatasetQualityReport } = vi.hoisted(() => ({
  getDatasetQualityReport: vi.fn(),
}));

vi.mock("../services/FileServices", () => ({ getDatasetQualityReport }));

import useDatasetQualityReport from "./useDatasetQualityReport";

beforeEach(() => {
  vi.useFakeTimers();
  getDatasetQualityReport.mockReset();
});

afterEach(() => vi.useRealTimers());

describe("useDatasetQualityReport", () => {
  it("debounces analysis and returns the current report", async () => {
    const report = { status: "clean", can_train: true };
    getDatasetQualityReport.mockResolvedValue(report);
    const { result } = renderHook(() => useDatasetQualityReport("file-1", "label", 2, 80));

    expect(result.current.loading).toBe(true);
    expect(getDatasetQualityReport).not.toHaveBeenCalled();
    await act(async () => vi.advanceTimersByTimeAsync(300));
    expect(getDatasetQualityReport).toHaveBeenCalledWith(
      "file-1",
      "label",
      2,
      80,
      expect.any(AbortSignal),
    );
    expect(result.current.report).toEqual(report);
    expect(result.current.loading).toBe(false);
  });

  it("aborts stale requests when the target changes and ignores stale results", async () => {
    let resolveFirst;
    getDatasetQualityReport.mockImplementationOnce((_file, target, _task, _split, signal) => {
      return new Promise((resolve) => {
        resolveFirst = () => resolve({ status: "blocked", target });
        signal.addEventListener("abort", () => {});
      });
    });
    getDatasetQualityReport.mockResolvedValueOnce({ status: "clean", target: "new_target" });
    const { result, rerender } = renderHook(
      ({ target }) => useDatasetQualityReport("file-1", target, 1, 80),
      { initialProps: { target: "old_target" } },
    );

    await act(async () => vi.advanceTimersByTimeAsync(300));
    const oldSignal = getDatasetQualityReport.mock.calls[0][4];
    rerender({ target: "new_target" });
    expect(oldSignal.aborted).toBe(true);
    await act(async () => vi.advanceTimersByTimeAsync(300));
    await act(async () => {
      resolveFirst();
      await Promise.resolve();
    });
    expect(result.current.report).toEqual({ status: "clean", target: "new_target" });
  });

  it("surfaces analysis failures and clears state when no file is selected", async () => {
    getDatasetQualityReport.mockRejectedValue(new Error("offline"));
    const { result, rerender } = renderHook(
      ({ fileId }) => useDatasetQualityReport(fileId, "label", 1, 80),
      { initialProps: { fileId: "file-1" } },
    );
    await act(async () => vi.advanceTimersByTimeAsync(300));
    expect(result.current.error).toBe("offline");

    rerender({ fileId: "" });
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.report).toBeNull();
  });
});
