import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import FeatureSelectionStudio from "./FeatureSelectionStudio";

vi.mock("../../services/FeatureSelectionService", () => ({
  analyzeFeatures: vi.fn(),
  applyFeatureSelection: vi.fn(),
}));

import { analyzeFeatures, applyFeatureSelection } from "../../services/FeatureSelectionService";

const profile = {
  target: "label",
  task: "classification",
  sample_rows: 10,
  scores: [
    {
      feature: "signal",
      mutual_information: 0.8,
      anova_f: 10,
      variance: 0.5,
      composite_score: 1,
    },
    {
      feature: "noise",
      mutual_information: 0.1,
      anova_f: 1,
      variance: 0.2,
      composite_score: 0,
    },
  ],
};

describe("FeatureSelectionStudio", () => {
  const updateFileList = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("requires a target before analysis", async () => {
    render(
      <FeatureSelectionStudio
        fileId="file-1"
        datasetRevision={0}
        columns={["signal", "label"]}
        updateFileList={updateFileList}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Analyze relevance" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Choose the target column");
    expect(analyzeFeatures).not.toHaveBeenCalled();
  });

  it("profiles features and applies the selected pruning with the target protected", async () => {
    analyzeFeatures.mockResolvedValue(profile);
    applyFeatureSelection.mockResolvedValue({
      data: { removed_columns: ["noise"], backup_file: "dataset.csv.bak" },
    });
    render(
      <FeatureSelectionStudio
        fileId="file-1"
        datasetRevision={0}
        columns={["signal", "noise", "label"]}
        updateFileList={updateFileList}
      />,
    );

    fireEvent.change(screen.getByLabelText("Target column"), { target: { value: "label" } });
    fireEvent.click(screen.getByRole("button", { name: "Analyze relevance" }));

    expect(await screen.findByText("Classification")).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /signal/ })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: "Remove label" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: "Remove noise" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove selected features" }));

    await waitFor(() => {
      expect(applyFeatureSelection).toHaveBeenCalledWith("file-1", "label", ["noise"]);
      expect(updateFileList).toHaveBeenCalledOnce();
    });
    expect(await screen.findByRole("status")).toHaveTextContent("dataset.csv.bak");
  });
});
