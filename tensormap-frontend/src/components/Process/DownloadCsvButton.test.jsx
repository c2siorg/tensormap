import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DownloadCsvButton from "./DownloadCsvButton";

const { downloadCsvFile } = vi.hoisted(() => ({ downloadCsvFile: vi.fn() }));

vi.mock("../../services/FileServices", () => ({ downloadCsvFile }));

describe("DownloadCsvButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.URL.createObjectURL = vi.fn(() => "blob:test-csv");
    window.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => vi.restoreAllMocks());

  it("downloads the returned CSV blob using the server filename", async () => {
    const user = userEvent.setup();
    const blob = new Blob(["a,b\n1,2\n"], { type: "text/csv" });
    downloadCsvFile.mockResolvedValue({ blob, filename: "processed.csv" });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<DownloadCsvButton fileId="csv-1" fileName="iris" fileType="csv" />);
    await user.click(screen.getByRole("button", { name: "Download CSV" }));

    await waitFor(() => expect(downloadCsvFile).toHaveBeenCalledWith("csv-1"));
    expect(window.URL.createObjectURL).toHaveBeenCalledWith(blob);
    expect(click).toHaveBeenCalled();
    await waitFor(() => expect(window.URL.revokeObjectURL).toHaveBeenCalledWith("blob:test-csv"));
  });

  it("uses a safe dataset-based filename when the response has no filename", async () => {
    const user = userEvent.setup();
    downloadCsvFile.mockResolvedValue({ blob: new Blob(["a,b\n"]), filename: null });
    let downloadName = "";
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName) => {
      const element = originalCreateElement(tagName);
      if (tagName === "a") {
        Object.defineProperty(element, "download", {
          set: (value) => {
            downloadName = value;
          },
        });
        element.click = vi.fn();
      }
      return element;
    });

    render(<DownloadCsvButton fileId="csv-1" fileName="../iris.csv" fileType="csv" />);
    await user.click(screen.getByRole("button", { name: "Download CSV" }));

    await waitFor(() => expect(downloadName).toBe(".._iris.csv"));
  });

  it("shows a helpful error when downloading fails", async () => {
    const user = userEvent.setup();
    downloadCsvFile.mockRejectedValue(new Error("network error"));

    render(<DownloadCsvButton fileId="csv-1" fileName="iris" fileType="csv" />);
    await user.click(screen.getByRole("button", { name: "Download CSV" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not download this CSV");
  });

  it("does not show an action for non-CSV datasets", () => {
    render(<DownloadCsvButton fileId="zip-1" fileName="images" fileType="zip" />);

    expect(screen.queryByRole("button", { name: "Download CSV" })).not.toBeInTheDocument();
    expect(downloadCsvFile).not.toHaveBeenCalled();
  });
});
