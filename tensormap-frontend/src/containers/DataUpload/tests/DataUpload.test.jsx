import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RecoilRoot } from "recoil";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import DataUpload from "../DataUpload";
import { getAllFiles, deleteFile, downloadDataset } from "../../../services/FileServices";

vi.mock("../../../services/FileServices", () => ({
  getAllFiles: vi.fn(),
  uploadFile: vi.fn(),
  deleteFile: vi.fn(),
  downloadDataset: vi.fn(() => Promise.resolve()),
}));

const SAMPLE_CSV = {
  file_id: "csv-123",
  file_name: "sample_data",
  file_type: "csv",
  fields: ["feature1", "feature2", "target"],
  row_count: 150,
};

function renderDataUpload() {
  return render(
    <RecoilRoot>
      <MemoryRouter initialEntries={["/workspace/p1/data-upload"]}>
        <Routes>
          <Route path="/workspace/:projectId/data-upload" element={<DataUpload />} />
        </Routes>
      </MemoryRouter>
    </RecoilRoot>,
  );
}

describe("DataUpload Container", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders empty state when no dataset is uploaded", async () => {
    getAllFiles.mockResolvedValue([]);
    renderDataUpload();

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Choose CSV File/i })).toBeTruthy();
    });
  });

  it("renders dataset details and download button when file exists", async () => {
    getAllFiles.mockResolvedValue([SAMPLE_CSV]);
    renderDataUpload();

    await waitFor(() => {
      expect(screen.getByText("sample_data.csv")).toBeTruthy();
      expect(screen.getByText(/Columns: 3/)).toBeTruthy();
      expect(screen.getByText(/Rows: 150/)).toBeTruthy();
      expect(screen.getByRole("button", { name: /Download/i })).toBeTruthy();
      expect(screen.getByRole("button", { name: /Replace/i })).toBeTruthy();
      expect(screen.getByRole("button", { name: /Remove/i })).toBeTruthy();
    });
  });

  it("triggers downloadDataset when Download button is clicked", async () => {
    const user = userEvent.setup();
    getAllFiles.mockResolvedValue([SAMPLE_CSV]);
    renderDataUpload();

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Download/i })).toBeTruthy();
    });

    const downloadBtn = screen.getByRole("button", { name: /Download/i });
    await user.click(downloadBtn);

    expect(downloadDataset).toHaveBeenCalledWith("csv-123", "sample_data.csv");
  });

  it("triggers deleteFile when Remove button is clicked", async () => {
    const user = userEvent.setup();
    getAllFiles.mockResolvedValue([SAMPLE_CSV]);
    deleteFile.mockResolvedValue({ success: true });
    renderDataUpload();

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Remove/i })).toBeTruthy();
    });

    const removeBtn = screen.getByRole("button", { name: /Remove/i });
    await user.click(removeBtn);

    expect(deleteFile).toHaveBeenCalledWith("csv-123");
  });
});
