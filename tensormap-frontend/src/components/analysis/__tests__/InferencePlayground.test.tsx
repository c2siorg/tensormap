/**
 * Tests for InferencePlayground component.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import InferencePlayground from "../InferencePlayground";
import * as InferenceService from "@/services/InferenceService";
import type { ModelFeatureSchemaResponse, InferenceResponse } from "@/types/inference";

vi.mock("@/services/InferenceService");

describe("InferencePlayground", () => {
  const mockSchema: ModelFeatureSchemaResponse = {
    job_id: "test-job-123",
    target_field: "species",
    problem_type: "classification",
    features: [
      {
        name: "sepal_length",
        dtype: "float",
        min: 4.3,
        max: 7.9,
        mean: 5.84,
        median: 5.8,
        step: 0.1,
        default_value: 5.8,
      },
      {
        name: "petal_width",
        dtype: "float",
        min: 0.1,
        max: 2.5,
        mean: 1.2,
        median: 1.3,
        step: 0.1,
        default_value: 1.3,
      },
    ],
    class_names: ["Setosa", "Versicolor", "Virginica"],
    sample_records: [
      {
        index: 0,
        features: { sepal_length: 5.1, petal_width: 0.2 },
        actual_class: 0,
        actual_class_name: "Setosa",
      },
    ],
  };

  const mockPrediction: InferenceResponse = {
    success: true,
    problem_type: "classification",
    predicted_class: 0,
    predicted_label: "Setosa",
    confidence: 0.94,
    probabilities: [
      { class_index: 0, class_name: "Setosa", probability: 0.94 },
      { class_index: 1, class_name: "Versicolor", probability: 0.04 },
      { class_index: 2, class_name: "Virginica", probability: 0.02 },
    ],
    latency_ms: 5.4,
    input_features: { sepal_length: 5.8, petal_width: 1.3 },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders feature sliders and prediction output", async () => {
    vi.spyOn(InferenceService, "getFeatureSchema").mockResolvedValue(mockSchema);
    vi.spyOn(InferenceService, "runInference").mockResolvedValue(mockPrediction);

    render(<InferencePlayground jobId="test-job-123" />);

    expect(screen.getByText("Loading interactive inference playground...")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Input Features (2)")).toBeInTheDocument();
      expect(screen.getByText("sepal_length")).toBeInTheDocument();
      expect(screen.getByText("petal_width")).toBeInTheDocument();
      expect(screen.getByText("Predicted Class")).toBeInTheDocument();
      expect(screen.getAllByText("Setosa")[0]).toBeInTheDocument();
      expect(screen.getByText("94% Confidence")).toBeInTheDocument();
      expect(screen.getByText("Class Probability Distribution")).toBeInTheDocument();
    });
  });

  it("loads a random test sample and triggers prediction", async () => {
    vi.spyOn(InferenceService, "getFeatureSchema").mockResolvedValue(mockSchema);
    vi.spyOn(InferenceService, "runInference").mockResolvedValue(mockPrediction);

    render(<InferencePlayground jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Random Test Sample")).toBeInTheDocument();
    });

    const sampleButton = screen.getByText("Random Test Sample");
    fireEvent.click(sampleButton);

    await waitFor(() => {
      expect(InferenceService.runInference).toHaveBeenCalledTimes(2);
      expect(screen.getByText(/Sample #1 Match/)).toBeInTheDocument();
    });
  });

  it("resets feature values to medians", async () => {
    vi.spyOn(InferenceService, "getFeatureSchema").mockResolvedValue(mockSchema);
    vi.spyOn(InferenceService, "runInference").mockResolvedValue(mockPrediction);

    render(<InferencePlayground jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Reset Medians")).toBeInTheDocument();
    });

    const resetButton = screen.getByText("Reset Medians");
    fireEvent.click(resetButton);

    await waitFor(() => {
      expect(InferenceService.runInference).toHaveBeenCalledWith("test-job-123", {
        sepal_length: 5.8,
        petal_width: 1.3,
      });
    });
  });

  it("handles regression model predictions", async () => {
    const mockRegressionSchema: ModelFeatureSchemaResponse = {
      ...mockSchema,
      problem_type: "regression",
      target_field: "price",
      class_names: undefined,
    };

    const mockRegressionPred: InferenceResponse = {
      success: true,
      problem_type: "regression",
      predicted_value: 245.5,
      latency_ms: 4.8,
      input_features: { sepal_length: 5.8, petal_width: 1.3 },
    };

    vi.spyOn(InferenceService, "getFeatureSchema").mockResolvedValue(mockRegressionSchema);
    vi.spyOn(InferenceService, "runInference").mockResolvedValue(mockRegressionPred);

    render(<InferencePlayground jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Predicted Value")).toBeInTheDocument();
      expect(screen.getByText("245.5")).toBeInTheDocument();
      expect(screen.getByText("Target Field: price")).toBeInTheDocument();
    });
  });

  it("displays error message on schema failure", async () => {
    vi.spyOn(InferenceService, "getFeatureSchema").mockRejectedValue(
      new Error("Inference only available for completed jobs"),
    );

    render(<InferencePlayground jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Inference only available for completed jobs")).toBeInTheDocument();
    });
  });
});
