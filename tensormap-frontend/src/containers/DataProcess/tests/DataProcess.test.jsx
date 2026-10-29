import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RecoilRoot } from "recoil";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import DataProcess from "../DataProcess";
import { projectFiles } from "../../../shared/atoms";

vi.mock("../../../services/FileServices", () => ({
  getAllFiles: vi.fn(),
}));

vi.mock("../../../components/Process/SelectFileModal", () => ({
  default: () => <div data-testid="select-file-modal">Select File Modal</div>,
}));

vi.mock("../../../components/Process/DisplayDataset", () => ({
  default: ({ fileId }) => <div data-testid="display-dataset">Display Dataset: {fileId}</div>,
}));

vi.mock("../../../components/Process/PreprocessData", () => ({
  default: ({ fileId }) => <div data-testid="preprocess-data">Preprocess Data: {fileId}</div>,
}));

vi.mock("../../../components/Process/ColumnStatsPanel", () => ({
  default: ({ fileId }) => <div data-testid="column-stats-panel">Column Stats: {fileId}</div>,
}));

vi.mock("../../../components/Process/FeatureDistributionPanel", () => ({
  default: ({ fileId }) => (
    <div data-testid="feature-distribution-panel">Feature Distribution: {fileId}</div>
  ),
}));

vi.mock("../../../components/Process/CorrelationHeatmap", () => ({
  default: ({ fileId }) => (
    <div data-testid="correlation-heatmap">Correlation Heatmap: {fileId}</div>
  ),
}));

const CSV_FILE = {
  file_id: "csv-1",
  file_name: "iris.csv",
  file_type: "csv",
};

const SECOND_CSV_FILE = {
  file_id: "csv-2",
  file_name: "housing.csv",
  file_type: "csv",
};

const ZIP_FILE = {
  file_id: "zip-1",
  file_name: "mnist.zip",
  file_type: "zip",
};

function renderDataProcess(initialFiles = []) {
  return render(
    <RecoilRoot
      initializeState={({ set }) => {
        set(projectFiles, initialFiles);
      }}
    >
      <MemoryRouter initialEntries={["/workspace/p1/data-process"]}>
        <Routes>
          <Route path="/workspace/:projectId/data-process" element={<DataProcess />} />
        </Routes>
      </MemoryRouter>
    </RecoilRoot>,
  );
}

describe("DataProcess Container", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders SelectFileModal when no files exist", async () => {
    renderDataProcess([]);

    await waitFor(() => {
      expect(screen.getByTestId("select-file-modal")).toBeTruthy();
    });
  });

  it("auto-selects first CSV file and renders tabular processing panels", async () => {
    // Pass ZIP file first in array to verify preference for CSV
    renderDataProcess([ZIP_FILE, CSV_FILE]);

    await waitFor(() => {
      expect(screen.getByTestId("display-dataset")).toBeTruthy();
      expect(screen.getByTestId("preprocess-data")).toBeTruthy();
      expect(screen.getByTestId("column-stats-panel")).toBeTruthy();
      expect(screen.getByTestId("feature-distribution-panel")).toBeTruthy();
      expect(screen.getByTestId("correlation-heatmap")).toBeTruthy();
    });

    expect(screen.getByText("Display Dataset: csv-1")).toBeTruthy();
    expect(screen.getByText("Preprocess Data: csv-1")).toBeTruthy();

    const select = screen.getByLabelText("Select Dataset:");
    expect(select.value).toBe("csv-1");
  });

  it("displays informative warning card when an image dataset (ZIP) is selected", async () => {
    const user = userEvent.setup();
    renderDataProcess([CSV_FILE, ZIP_FILE]);

    await waitFor(() => {
      expect(screen.getByTestId("display-dataset")).toBeTruthy();
    });

    const select = screen.getByLabelText("Select Dataset:");
    await user.selectOptions(select, "zip-1");

    await waitFor(() => {
      expect(screen.getByText("Image Dataset Selected")).toBeTruthy();
      expect(screen.getAllByText(/mnist.zip/).length).toBeGreaterThanOrEqual(1);
      expect(
        screen.getByText(/Tabular operations.*only apply to tabular CSV datasets/),
      ).toBeTruthy();
    });

    // Verify tabular panels are not mounted for the zip file
    expect(screen.queryByTestId("display-dataset")).toBeNull();
    expect(screen.queryByTestId("column-stats-panel")).toBeNull();
  });

  it("allows switching between multiple CSV datasets", async () => {
    const user = userEvent.setup();
    renderDataProcess([CSV_FILE, SECOND_CSV_FILE]);

    await waitFor(() => {
      expect(screen.getByText("Display Dataset: csv-1")).toBeTruthy();
    });

    const select = screen.getByLabelText("Select Dataset:");
    await user.selectOptions(select, "csv-2");

    await waitFor(() => {
      expect(screen.getByText("Display Dataset: csv-2")).toBeTruthy();
      expect(screen.getByText("Preprocess Data: csv-2")).toBeTruthy();
    });
  });

  it("formats dropdown options with image dataset tag for non-csv files", async () => {
    renderDataProcess([CSV_FILE, ZIP_FILE]);

    await waitFor(() => {
      const options = screen.getAllByRole("option");
      // Option 0: disabled placeholder "Choose a dataset"
      // Option 1: CSV file "iris.csv"
      // Option 2: ZIP file "mnist.zip (ZIP - Image Dataset)"
      expect(options[1].textContent).toBe("iris.csv");
      expect(options[2].textContent).toBe("mnist.zip (ZIP - Image Dataset)");
    });
  });
});
