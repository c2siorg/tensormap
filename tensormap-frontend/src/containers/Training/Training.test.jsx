import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RecoilRoot } from "recoil";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Training from "./Training";

vi.mock("../../services/ModelServices", () => ({
  download_code: vi.fn(),
  runModel: vi.fn(),
  getTrainingHistory: vi.fn(),
  updateTrainingConfig: vi.fn(),
  deleteModel: vi.fn(),
}));

vi.mock("../../services/FileServices", () => ({
  getAllFiles: vi.fn(),
}));

vi.mock("../../services/socketService", () => ({
  getTrainingSocket: () => ({
    connected: true,
    on: vi.fn(),
    off: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
  }),
}));

vi.mock("@/components/training/TrainingMetricsChart", () => ({ default: () => null }));
vi.mock("@/components/export/ExportPanel", () => ({ default: () => null }));

import { getTrainingHistory, updateTrainingConfig } from "../../services/ModelServices";
import { getAllFiles } from "../../services/FileServices";

const SAVED_MODEL = { id: 1, model_name: "demo", created_on: "2026-01-01T00:00:00Z" };
const CSV_FILE = { file_id: "file-1", file_name: "iris", file_type: "csv", fields: ["sepal"] };

function renderTraining() {
  return render(
    <RecoilRoot>
      <MemoryRouter initialEntries={["/workspace/p1/training"]}>
        <Routes>
          <Route path="/workspace/:projectId/training" element={<Training />} />
        </Routes>
      </MemoryRouter>
    </RecoilRoot>,
  );
}

// Radix Select triggers carry no accessible name from <Label>, so find them
// through the label text in the same field wrapper.
function selectTrigger(labelText) {
  return within(screen.getByText(labelText).parentElement).getByRole("combobox");
}

async function pickOption(user, labelText, optionText) {
  await user.click(selectTrigger(labelText));
  await user.click(await screen.findByRole("option", { name: optionText }));
}

async function selectSavedModel(user) {
  await user.click(await screen.findByRole("combobox"));
  await user.click(await screen.findByRole("option", { name: /demo/ }));
  await screen.findByText("Loss Function");
}

describe("Training page loss function", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getTrainingHistory.mockResolvedValue([SAVED_MODEL]);
    getAllFiles.mockResolvedValue([CSV_FILE]);
    updateTrainingConfig.mockResolvedValue({ success: true });
  });

  it("groups the loss options under Classification and Regression", async () => {
    const user = userEvent.setup();
    renderTraining();
    await selectSavedModel(user);

    await user.click(selectTrigger("Loss Function"));

    const listbox = await screen.findByRole("listbox");
    const groups = within(listbox).getAllByRole("group");
    expect(groups).toHaveLength(2);
    expect(groups[0]).toHaveTextContent("Classification");
    expect(
      within(groups[0])
        .getAllByRole("option")
        .map((o) => o.textContent),
    ).toEqual([
      "Sparse Categorical Crossentropy",
      "Categorical Crossentropy",
      "Binary Crossentropy",
    ]);
    expect(groups[1]).toHaveTextContent("Regression");
    expect(
      within(groups[1])
        .getAllByRole("option")
        .map((o) => o.textContent),
    ).toEqual(["Mean Squared Error", "Mean Absolute Error", "Huber"]);
  });

  it("preselects the default loss for the chosen problem type", async () => {
    const user = userEvent.setup();
    renderTraining();
    await selectSavedModel(user);

    expect(selectTrigger("Loss Function")).toHaveTextContent("Select loss function");

    await pickOption(user, "Problem Type", "Multi class classification");
    expect(selectTrigger("Loss Function")).toHaveTextContent("Sparse Categorical Crossentropy");

    await pickOption(user, "Problem Type", "Linear Regression");
    expect(selectTrigger("Loss Function")).toHaveTextContent("Mean Squared Error");
  });

  it("lets the user override the default and sends it when saving", async () => {
    const user = userEvent.setup();
    renderTraining();
    await selectSavedModel(user);

    await pickOption(user, "Dataset File", "iris.csv");
    await pickOption(user, "Problem Type", "Linear Regression");
    await pickOption(user, "Loss Function", "Huber");
    expect(selectTrigger("Loss Function")).toHaveTextContent("Huber");

    await pickOption(user, "Result Metrics", "MSE");
    await user.type(screen.getByPlaceholderText("Select from list or enter field name"), "sepal");
    await user.type(screen.getByPlaceholderText("Number of epochs"), "3");
    await user.type(screen.getByPlaceholderText("Batch size"), "16");
    await user.type(screen.getByPlaceholderText("e.g. 0.8"), "0.8");

    await user.click(screen.getByRole("button", { name: "Save Configuration" }));

    await waitFor(() => expect(updateTrainingConfig).toHaveBeenCalledTimes(1));
    expect(updateTrainingConfig.mock.calls[0][0]).toMatchObject({
      model_name: "demo",
      problem_type_id: 2,
      loss: "huber",
      metric: "mse",
    });
  });
});
