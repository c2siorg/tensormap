import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PreprocessData from "../PreprocessData";
import { downloadDataset, getCovMatrix } from "../../../services/FileServices";

vi.mock("../../../services/FileServices", () => ({
  downloadDataset: vi.fn(),
  getCovMatrix: vi.fn(),
  transformData: vi.fn(),
}));

vi.mock("../TransformationCreator", () => ({ default: () => null }));
vi.mock("../TransformationList", () => ({ default: () => null }));
vi.mock("../../shared/FeedbackDialog", () => ({ default: () => null }));

describe("PreprocessData dataset downloads", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getCovMatrix.mockResolvedValue({ data_types: {} });
  });

  it("downloads the CSV using the response filename", async () => {
    let downloadLink;
    const createElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName, options) => {
      const element = createElement(tagName, options);
      if (tagName === "a") downloadLink = element;
      return element;
    });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    vi.spyOn(window.URL, "createObjectURL").mockReturnValue("blob:test-dataset");
    vi.spyOn(window.URL, "revokeObjectURL").mockImplementation(() => {});
    downloadDataset.mockResolvedValue({
      data: new Blob(["a,b\n1,2\n"], { type: "text/csv" }),
      headers: { "content-disposition": "attachment; filename*=UTF-8''iris%20export.csv" },
    });

    render(<PreprocessData fileId="dataset-1" fileName="iris" updateFileList={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));

    await waitFor(() => {
      expect(downloadDataset).toHaveBeenCalledWith("dataset-1");
      expect(downloadLink.download).toBe("iris export.csv");
      expect(downloadLink.click).toHaveBeenCalled();
      expect(window.URL.createObjectURL).toHaveBeenCalled();
    });
  });

  it("shows a clear error when the download fails", async () => {
    downloadDataset.mockRejectedValue({
      message: "Request failed",
      response: {
        data: new Blob([JSON.stringify({ detail: "Dataset file not found" })], {
          type: "application/json",
        }),
      },
    });
    render(<PreprocessData fileId="dataset-1" fileName="iris" updateFileList={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Dataset file not found");
  });
});
