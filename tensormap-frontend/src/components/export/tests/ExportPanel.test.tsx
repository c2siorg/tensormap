/**
 * Test suite for ExportPanel component.
 * @module
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ExportPanel from "../ExportPanel";
import axios from "@/shared/Axios";
import * as urls from "@/constants/Urls";

vi.mock("@/shared/Axios");

describe("ExportPanel", () => {
  const mockJobId = "job-abc-123";
  const mockModelName = "mnist_classifier";

  const mockFormatsResponse = {
    data: {
      formats: {
        savedmodel: {
          available: true,
          size_bytes: 1048576, // 1 MB
          expires_at: new Date(Date.now() + 86400000 * 5).toISOString(), // 5 days from now
        },
        tflite: {
          available: false,
          size_bytes: null,
          expires_at: null,
        },
        onnx: {
          available: false,
          size_bytes: null,
          expires_at: null,
          onnx_supported: false,
          onnx_issues: ["CustomLayer is not supported in ONNX runtime"],
        },
      },
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    window.URL.createObjectURL = vi.fn(() => "blob:http://localhost/test-blob-id");
    window.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows loading state while fetching formats", () => {
    (axios.get as any).mockReturnValue(new Promise(() => {}));

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    expect(screen.getByText("Loading export formats...")).toBeTruthy();
  });

  it("displays error and retry button when format fetch fails", async () => {
    const user = userEvent.setup();
    (axios.get as any).mockRejectedValueOnce({
      response: { data: { detail: "Database connection failed" } },
    });

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("Database connection failed")).toBeTruthy();
    });

    const retryButton = screen.getByRole("button", { name: /Retry/i });
    expect(retryButton).toBeTruthy();

    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);
    await user.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText("SavedModel")).toBeTruthy();
    });
  });

  it("renders formats, sizes, and expiry when loaded", async () => {
    (axios.get as any).mockResolvedValue(mockFormatsResponse);

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("SavedModel")).toBeTruthy();
      expect(screen.getByText("TFLite")).toBeTruthy();
      expect(screen.getByText("ONNX")).toBeTruthy();
    });

    // SavedModel is available
    expect(screen.getByText("Size: 1 MB")).toBeTruthy();
    expect(screen.getByText(/Expires in \d+ days/)).toBeTruthy();

    const buttons = screen.getAllByRole("button");
    // SavedModel has "Download", TFLite has "Generate & Download"
    expect(buttons[0].textContent).toContain("Download");
    expect(buttons[1].textContent).toContain("Generate & Download");
  });

  it("disables ONNX download button and shows issue alert when unsupported", async () => {
    (axios.get as any).mockResolvedValue(mockFormatsResponse);

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("CustomLayer is not supported in ONNX runtime")).toBeTruthy();
    });

    const onnxButton = screen.getAllByRole("button")[2];
    expect(onnxButton).toBeDisabled();
  });

  it("downloads available format with Content-Disposition filename", async () => {
    const user = userEvent.setup();
    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);

    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("SavedModel")).toBeTruthy();
    });

    const downloadBlob = new Blob(["saved-model-binary-data"], { type: "application/zip" });
    (axios.get as any).mockResolvedValueOnce({
      data: downloadBlob,
      headers: {
        "content-disposition": 'attachment; filename="custom_model_archive.zip"',
      },
    });
    // mock next fetchFormats call
    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);

    const downloadButton = screen.getAllByRole("button")[0];
    await user.click(downloadButton);

    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith(
        `${urls.BACKEND_MODEL_EXPORT}/${mockJobId}`,
        expect.objectContaining({
          params: { format: "savedmodel" },
          responseType: "blob",
        }),
      );
      expect(window.URL.createObjectURL).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();
      expect(window.URL.revokeObjectURL).toHaveBeenCalledWith("blob:http://localhost/test-blob-id");
    });

    clickSpy.mockRestore();
  });

  it("uses fallback filename when Content-Disposition is missing", async () => {
    const user = userEvent.setup();
    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);

    let createdDownloadAttr = "";
    const origCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
      const el = origCreateElement(tagName);
      if (tagName === "a") {
        const origSetAttribute = el.setAttribute.bind(el);
        el.setAttribute = (name: string, value: string) => {
          if (name === "download") {
            createdDownloadAttr = value;
          }
          return origSetAttribute(name, value);
        };
        el.click = vi.fn();
      }
      return el;
    });

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("TFLite")).toBeTruthy();
    });

    (axios.get as any).mockResolvedValueOnce({
      data: new Blob(["tflite-bytes"]),
      headers: {},
    });
    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);

    const tfliteButton = screen.getAllByRole("button")[1];
    await user.click(tfliteButton);

    await waitFor(() => {
      expect(createdDownloadAttr).toBe("mnist_classifier.tflite");
    });

    (document.createElement as any).mockRestore();
  });

  it("displays ONNX unsupported error parsed from blob response", async () => {
    const user = userEvent.setup();
    const formatsWithOnnxSupported = {
      data: {
        formats: {
          savedmodel: { available: true, size_bytes: 100, expires_at: null },
          tflite: { available: true, size_bytes: 100, expires_at: null },
          onnx: { available: false, size_bytes: null, expires_at: null, onnx_supported: true },
        },
      },
    };

    (axios.get as any).mockResolvedValueOnce(formatsWithOnnxSupported);

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("ONNX")).toBeTruthy();
    });

    const errorBlob = new Blob(
      [
        JSON.stringify({
          error: "onnx_unsupported",
          issues: ["Unsupported activation GELU"],
          suggestion: "Use ReLU activation instead",
        }),
      ],
      { type: "application/json" },
    );

    (axios.get as any).mockRejectedValueOnce({
      response: { data: errorBlob },
    });

    const onnxButton = screen.getAllByRole("button")[2];
    await user.click(onnxButton);

    await waitFor(() => {
      expect(
        screen.getByText(
          "ONNX export not supported: Unsupported activation GELU. Use ReLU activation instead",
        ),
      ).toBeTruthy();
    });
  });

  it("extracts detail message from server error blob response", async () => {
    const user = userEvent.setup();
    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("SavedModel")).toBeTruthy();
    });

    const errorBlob = new Blob(
      [JSON.stringify({ detail: "Export generation failed due to corrupted weights" })],
      { type: "application/json" },
    );

    (axios.get as any).mockRejectedValueOnce({
      response: { data: errorBlob },
    });

    const savedModelButton = screen.getAllByRole("button")[0];
    await user.click(savedModelButton);

    await waitFor(() => {
      expect(screen.getByText("Export generation failed due to corrupted weights")).toBeTruthy();
    });
  });

  it("handles general error when response data is not a blob", async () => {
    const user = userEvent.setup();
    (axios.get as any).mockResolvedValueOnce(mockFormatsResponse);

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("SavedModel")).toBeTruthy();
    });

    (axios.get as any).mockRejectedValueOnce(new Error("Network timeout"));

    const savedModelButton = screen.getAllByRole("button")[0];
    await user.click(savedModelButton);

    await waitFor(() => {
      expect(screen.getByText("Network timeout")).toBeTruthy();
    });
  });

  it("formats zero bytes and null expiry correctly", async () => {
    const zeroByteFormats = {
      data: {
        formats: {
          savedmodel: {
            available: true,
            size_bytes: 0,
            expires_at: null,
          },
          tflite: {
            available: true,
            size_bytes: null,
            expires_at: new Date(Date.now() - 86400000).toISOString(), // expired yesterday
          },
          onnx: {
            available: true,
            size_bytes: 2048,
            expires_at: new Date().toISOString(), // today
            onnx_supported: true,
          },
        },
      },
    };

    (axios.get as any).mockResolvedValue(zeroByteFormats);

    render(<ExportPanel jobId={mockJobId} modelName={mockModelName} />);

    await waitFor(() => {
      expect(screen.getByText("Size: 0 B")).toBeTruthy();
      expect(screen.getByText("Unknown")).toBeTruthy();
      expect(screen.getByText("Expired")).toBeTruthy();
      expect(screen.getByText("Expires today")).toBeTruthy();
      expect(screen.getByText("Size: 2 KB")).toBeTruthy();
    });
  });
});
