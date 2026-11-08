/**
 * Test suite for ReproducibilityPanel component.
 * @module
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ReproducibilityPanel from "../ReproducibilityPanel";
import * as ModelServices from "@/services/ModelServices";
import type { TrainingJobDetails, ReproducibilitySnapshot } from "@/types/reproducibility";

// Mock ModelServices
vi.mock("@/services/ModelServices", () => ({
  getTrainingJobDetails: vi.fn(),
}));

// Mock logger
vi.mock("@/shared/logger", () => ({
  default: {
    error: vi.fn(),
  },
}));

const createMockSnapshot = (): ReproducibilitySnapshot => ({
  schema_version: 2,
  job_id: "test-job-123",
  captured_at: "2026-11-03T12:00:00Z",
  dataset: {
    file_id: "file-123",
    filename: "iris.csv",
    dataset_type: "csv",
    fingerprint_algorithm: "sha256",
    fingerprint: "a1b2c3d4e5f6" + "0".repeat(52), // 64 char hex
    fingerprint_scope: "file_bytes",
    size_bytes: 123456,
  },
  training_configuration: {
    problem_type_id: 1,
    problem_type: "multi_class_classification",
    target_column: "species",
    feature_columns: ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    class_names: ["setosa", "versicolor", "virginica"],
    model_architecture: {},
    architecture_sha256: "def456",
    optimizer: { class_name: "Adam" },
    loss: { class_name: "SparseCategoricalCrossentropy" },
    metrics: ["accuracy"],
    epochs: 50,
    batch_size: 32,
    training_split_percent: 80.0,
    holdout_split_percent: 20.0,
    requested_training_split_percent: 80.0,
    requested_holdout_split_percent: 20.0,
    actual_training_rows: 120,
    actual_holdout_rows: 30,
    actual_training_split_percent: 80.0,
    actual_holdout_split_percent: 20.0,
    holdout_role: "evaluation",
    configured_hyperparameters: {},
    effective_hyperparameters: {},
  },
  randomness: {
    configured_seed: null,
    effective_seeds: {
      tabular_dataframe_sample_random_state: 42,
    },
    applied_seeds: ["tabular_dataframe_sample_random_state"],
    global_framework_seed_applied: false,
    deterministic_execution_guaranteed: false,
  },
  runtime_versions: {
    python: "3.12.13",
    tensorflow: "2.16.0",
    keras: "3.0.0",
    numpy: "1.24.3",
    pandas: "2.1.4",
  },
});

const createMockJobDetails = (
  reproducibility_metadata: ReproducibilitySnapshot | null = null,
): TrainingJobDetails => ({
  job_id: "test-job-123",
  model_id: 1,
  status: "completed",
  hyperparams: {},
  started_at: "2026-11-03T12:00:00Z",
  completed_at: "2026-11-03T12:30:00Z",
  error_message: null,
  latest_metrics: {},
  reproducibility_metadata,
});

describe("ReproducibilityPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders collapsed by default", () => {
    render(<ReproducibilityPanel jobId="test-job-123" />);
    expect(screen.getByText("Run Details")).toBeTruthy();
    expect(screen.queryByText("Dataset")).toBeNull();
  });

  it("expands when header is clicked", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText("Dataset")).toBeTruthy();
    });
  });

  it("collapses when header is clicked again", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });

    // Expand
    await user.click(header);
    await waitFor(() => {
      expect(screen.getByText("Dataset")).toBeTruthy();
    });

    // Collapse
    await user.click(header);
    await waitFor(() => {
      expect(screen.queryByText("Dataset")).toBeNull();
    });
  });

  it("shows loading state while fetching", async () => {
    const user = userEvent.setup();
    let resolvePromise: any;
    const promise = new Promise<TrainingJobDetails>((resolve) => {
      resolvePromise = resolve;
    });
    vi.mocked(ModelServices.getTrainingJobDetails).mockReturnValue(promise);

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    expect(screen.getByText(/loading run details/i)).toBeTruthy();

    // Resolve the promise
    resolvePromise(createMockJobDetails(createMockSnapshot()));

    await waitFor(() => {
      expect(screen.queryByText(/loading run details/i)).toBeNull();
    });
  });

  it("displays all metadata sections when available", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText("Dataset")).toBeTruthy();
      expect(screen.getByText("Training Configuration")).toBeTruthy();
      expect(screen.getByText("Data Split")).toBeTruthy();
      expect(screen.getByText("Randomness")).toBeTruthy();
      expect(screen.getByText("Runtime Versions")).toBeTruthy();
    });
  });

  it("displays dataset fingerprint with copy button", async () => {
    const user = userEvent.setup();
    const mockData = createMockJobDetails(createMockSnapshot());
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(mockData);

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/a1b2c3d4e5f6/)).toBeTruthy();
    });

    const copyButton = screen.getByRole("button", { name: /copy fingerprint/i });
    expect(copyButton).toBeTruthy();
  });

  it("copies fingerprint to clipboard", async () => {
    const user = userEvent.setup();
    const mockData = createMockJobDetails(createMockSnapshot());
    const fingerprint = mockData.reproducibility_metadata!.dataset.fingerprint;
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(mockData);

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/a1b2c3d4e5f6/)).toBeTruthy();
    });

    // Spy on clipboard after render
    const writeTextSpy = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();

    const copyButton = screen.getByRole("button", { name: /copy fingerprint/i });
    await user.click(copyButton);

    expect(writeTextSpy).toHaveBeenCalledWith(fingerprint);
    writeTextSpy.mockRestore();
  });

  it("shows unavailable message for legacy jobs", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(null), // No metadata
    );

    render(<ReproducibilityPanel jobId="test-job-123" jobStatus="completed" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/reproducibility metadata is unavailable/i)).toBeTruthy();
      expect(screen.getByText(/this feature was added after this job was created/i)).toBeTruthy();
    });
  });

  it("shows pending message for active jobs without metadata", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(null), // No metadata yet
    );

    render(<ReproducibilityPanel jobId="test-job-123" jobStatus="running" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/run details will be available/i)).toBeTruthy();
      expect(
        screen.queryByText(/this feature was added after this job was created/i),
      ).toBeNull();
    });
  });

  it("displays missing optional runtime versions as unavailable", async () => {
    const user = userEvent.setup();
    const mockData = createMockJobDetails(createMockSnapshot());
    mockData.reproducibility_metadata!.runtime_versions.tensorflow = null;
    mockData.reproducibility_metadata!.runtime_versions.keras = null;
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(mockData);

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      const unavailableTexts = screen.getAllByText("Not available");
      expect(unavailableTexts.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("handles API failure and shows retry button", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockRejectedValue(
      new Error("Network error"),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/network error/i)).toBeTruthy();
      expect(screen.getByRole("button", { name: /retry/i })).toBeTruthy();
    });
  });

  it("retries fetching on retry button click", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockRejectedValueOnce(
      new Error("Network error"),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/network error/i)).toBeTruthy();
    });

    // Setup successful response for retry
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    const retryButton = screen.getByRole("button", { name: /retry/i });
    await user.click(retryButton);

    await waitFor(() => {
      expect(screen.queryByText(/network error/i)).toBeNull();
      expect(screen.getByText("Dataset")).toBeTruthy();
    });
  });

  it("fetches new data when job ID changes", async () => {
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    const { rerender } = render(<ReproducibilityPanel jobId="job-1" />);

    expect(ModelServices.getTrainingJobDetails).toHaveBeenCalledWith("job-1");

    rerender(<ReproducibilityPanel jobId="job-2" />);

    await waitFor(() => {
      expect(ModelServices.getTrainingJobDetails).toHaveBeenCalledWith("job-2");
    });
  });

  it("formats byte sizes correctly", async () => {
    const user = userEvent.setup();
    const mockData = createMockJobDetails(createMockSnapshot());
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(mockData);

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      // 123456 bytes = 120.56 KB
      expect(screen.getByText(/120\.56 KB/i)).toBeTruthy();
    });
  });

  it("shows deterministic execution warning", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/deterministic execution is not guaranteed/i)).toBeTruthy();
      expect(screen.getByText(/tensorflow gpu operations/i)).toBeTruthy();
    });
  });

  it("displays captured timestamp in readable format", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/captured:/i)).toBeTruthy();
    });
  });

  it("displays schema version", async () => {
    const user = userEvent.setup();
    vi.mocked(ModelServices.getTrainingJobDetails).mockResolvedValue(
      createMockJobDetails(createMockSnapshot()),
    );

    render(<ReproducibilityPanel jobId="test-job-123" />);

    const header = screen.getByRole("button", { name: /run details/i });
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText(/schema version: 2/i)).toBeTruthy();
    });
  });
});
