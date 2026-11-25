import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

describe("useLayerRegistry", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  it("fetches layers from the shared backend base_URL (port 4300 default) rather than port 5000", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        {
          type_key: "dense",
          display_name: "Dense",
          category: "core",
          keras_class: "tf.keras.layers.Dense",
          merge: false,
          description: "Dense layer",
          params: {},
        },
      ],
    });
    vi.stubGlobal("fetch", fetchMock);

    const { useLayerRegistry } = await import("../useLayerRegistry");
    const { result } = renderHook(() => useLayerRegistry());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const requestedUrl = String(fetchMock.mock.calls[0][0]);
    expect(requestedUrl).toContain(":4300/api/v1/layers");
    expect(requestedUrl).not.toContain(":5000/");
  });
});
