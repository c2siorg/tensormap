import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Tuning from "../Tuning";

vi.mock("@/services/ModelServices", () => ({
  getTrainingHistory: vi.fn(),
}));

vi.mock("@/components/tuning", () => ({
  TuningContainer: function MockTuningContainer({ modelName }) {
    const [mountToken] = useState(() => Math.random().toString(36).slice(2));
    return (
      <div data-testid="tuning-container" data-mount-token={mountToken}>
        TuningContainer: {modelName}
      </div>
    );
  },
}));

import { getTrainingHistory } from "@/services/ModelServices";

function renderTuning(initialUrl = "/workspace/proj-1/tuning") {
  return render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <Routes>
        <Route path="/workspace/:projectId/tuning" element={<Tuning />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Tuning Container Page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders empty state when project has no models", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([]);

    renderTuning();

    await waitFor(() => {
      expect(screen.getByText(/No saved models yet/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Go to Models Canvas/i })).toBeInTheDocument();
    });
  });

  it("shows training configuration warning when baseline config is missing", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([
      {
        id: 1,
        model_name: "unconfigured_net",
        epochs: null,
        training_split: null,
        optimizer: null,
        batch_size: null,
        loss: null,
      },
    ]);

    renderTuning();

    await waitFor(() => {
      expect(screen.getByText(/Training configuration required/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Configure Training/i })).toBeInTheDocument();
    });
    expect(screen.queryByTestId("tuning-container")).not.toBeInTheDocument();
  });

  it("mounts TuningContainer for configured model and honors ?model= param", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([
      {
        id: 1,
        model_name: "model_alpha",
        epochs: 10,
        training_split: 80,
        optimizer: "adam",
        batch_size: 32,
        loss: "sparse_categorical_crossentropy",
      },
      {
        id: 2,
        model_name: "model_beta",
        epochs: 25,
        training_split: 75,
        optimizer: "sgd",
        batch_size: 64,
        loss: "mean_squared_error",
      },
    ]);

    renderTuning("/workspace/proj-1/tuning?model=model_beta");

    await waitFor(() => {
      expect(screen.getByTestId("tuning-container")).toHaveTextContent(
        "TuningContainer: model_beta",
      );
      expect(screen.getByText(/Optimizer: sgd/i)).toBeInTheDocument();
      expect(screen.getByText(/Batch size: 64/i)).toBeInTheDocument();
      expect(screen.getByText(/Epochs: 25/i)).toBeInTheDocument();
    });
  });

  it("displays an error alert if loading models fails", async () => {
    vi.mocked(getTrainingHistory).mockRejectedValue(new Error("Network error"));

    renderTuning();

    await waitFor(() => {
      expect(screen.getByText(/Failed to load saved models/i)).toBeInTheDocument();
    });
  });

  it("shows training configuration warning when loss or optimizer is unset", async () => {
    vi.mocked(getTrainingHistory).mockResolvedValue([
      {
        id: 1,
        model_name: "partial_net",
        epochs: 10,
        training_split: 80,
        optimizer: "adam",
        batch_size: 32,
        loss: null,
      },
    ]);

    renderTuning();

    await waitFor(() => {
      expect(screen.getByText(/Training configuration required/i)).toBeInTheDocument();
    });
    expect(screen.queryByTestId("tuning-container")).not.toBeInTheDocument();
  });

  it("does not reload models when changing selection and keeps TuningContainer mounted on refresh", async () => {
    const user = userEvent.setup();
    vi.mocked(getTrainingHistory).mockResolvedValue([
      {
        id: 1,
        model_name: "model_alpha",
        epochs: 10,
        training_split: 80,
        optimizer: "adam",
        batch_size: 32,
        loss: "sparse_categorical_crossentropy",
      },
      {
        id: 2,
        model_name: "model_beta",
        epochs: 25,
        training_split: 75,
        optimizer: "sgd",
        batch_size: 64,
        loss: "mean_squared_error",
      },
    ]);

    renderTuning();

    await waitFor(() => {
      expect(screen.getByTestId("tuning-container")).toHaveTextContent(
        "TuningContainer: model_alpha",
      );
    });
    const initialToken = screen.getByTestId("tuning-container").getAttribute("data-mount-token");
    expect(getTrainingHistory).toHaveBeenCalledTimes(1);

    // Refreshing models must NOT unmount TuningContainer
    await user.click(screen.getByRole("button", { name: /Refresh Models/i }));
    await waitFor(() => {
      expect(getTrainingHistory).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByTestId("tuning-container").getAttribute("data-mount-token")).toBe(
      initialToken,
    );

    // Switching model updates TuningContainer via URL param WITHOUT re-fetching models
    await user.click(screen.getByRole("combobox", { name: /Select model to tune/i }));
    await user.click(screen.getByRole("option", { name: "model_beta" }));

    await waitFor(() => {
      expect(screen.getByTestId("tuning-container")).toHaveTextContent(
        "TuningContainer: model_beta",
      );
    });
    expect(screen.queryByText(/Loading models\.\.\./i)).not.toBeInTheDocument();
    expect(getTrainingHistory).toHaveBeenCalledTimes(2);
  });
});
