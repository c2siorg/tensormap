import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ReproducibilityDetails from "../ReproducibilityDetails";
import axios from "@/shared/Axios";

vi.mock("@/shared/Axios", () => ({
  default: {
    get: vi.fn(),
  },
}));

const mockAxiosGet = vi.mocked(axios.get);

const SAMPLE_SNAPSHOT = {
  captured_at: "2026-11-06T04:00:00Z",
  dataset: {
    file_id: "file-123",
    file_name: "credit_card.csv",
    file_type: "csv",
    disk_name: "credit_card_disk.csv",
    fingerprint: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    file_size_bytes: 204800,
    row_count: 5000,
  },
  training_config: {
    model_name: "fraud_detector",
    problem_type: "classification",
    problem_type_id: 1,
    target_field: "is_fraud",
    optimizer: "adam",
    learning_rate: 0.001,
    loss: "binary_crossentropy",
    metric: "accuracy",
    epochs: 25,
    batch_size: 64,
    training_split: 80,
    hyperparameters: { optimizer: "adam", lr: 0.001, epochs: 25, batch_size: 64 },
  },
  random_seed: 42,
  runtime_environment: {
    python_version: "3.11.9",
    platform: "Linux-6.1.0-custom",
    libraries: {
      tensorflow: "2.16.1",
      numpy: "1.26.4",
      pandas: "2.2.2",
    },
  },
};

describe("ReproducibilityDetails", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state initially", () => {
    mockAxiosGet.mockReturnValue(new Promise(() => {}));
    render(<ReproducibilityDetails jobId="job-abc-123" />);
    expect(screen.getByText(/loading reproducibility metadata/i)).toBeInTheDocument();
  });

  it("renders full reproducibility details when snapshot exists", async () => {
    mockAxiosGet.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          job_id: "job-abc-123",
          has_snapshot: true,
          reproducibility: SAMPLE_SNAPSHOT,
        },
      },
    });

    render(<ReproducibilityDetails jobId="job-abc-123" />);

    // Wait for content
    await waitFor(() => {
      expect(screen.getByText(/reproducibility & execution context/i)).toBeInTheDocument();
    });

    // Dataset checks
    expect(screen.getByText("credit_card.csv")).toBeInTheDocument();
    expect(screen.getByText("200 KB")).toBeInTheDocument();
    expect(screen.getByText("5,000")).toBeInTheDocument();
    expect(screen.getByText(/sha256:e3b0c44298fc1/i)).toBeInTheDocument();

    // Config checks
    expect(screen.getByText("classification")).toBeInTheDocument();
    expect(screen.getByText("is_fraud")).toBeInTheDocument();
    expect(screen.getByText("0.001")).toBeInTheDocument();
    expect(screen.getByText("25 / 64")).toBeInTheDocument();
    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();

    // Environment checks
    expect(screen.getByText("3.11.9")).toBeInTheDocument();
    expect(screen.getByText("2.16.1")).toBeInTheDocument();
    expect(screen.getByText("1.26.4")).toBeInTheDocument();
    expect(screen.getByText("2.2.2")).toBeInTheDocument();
  });

  it("displays fallback card for legacy jobs without snapshot", async () => {
    mockAxiosGet.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          job_id: "legacy-job-456",
          has_snapshot: false,
          reproducibility: null,
        },
      },
    });

    render(<ReproducibilityDetails jobId="legacy-job-456" />);

    await waitFor(() => {
      expect(screen.getByText(/no reproducibility snapshot recorded/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/legacy jobs executed prior to snapshot tracking/i)).toBeInTheDocument();
  });

  it("distinguishes unconfigured values from unavailable values", async () => {
    const partialSnapshot = {
      ...SAMPLE_SNAPSHOT,
      dataset: {
        ...SAMPLE_SNAPSHOT.dataset,
        fingerprint: null,
      },
      training_config: {
        ...SAMPLE_SNAPSHOT.training_config,
        learning_rate: null,
        target_field: null,
      },
      random_seed: null,
    };

    mockAxiosGet.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          job_id: "job-partial-789",
          has_snapshot: true,
          reproducibility: partialSnapshot,
        },
      },
    });

    render(<ReproducibilityDetails jobId="job-partial-789" />);

    await waitFor(() => {
      expect(screen.getByText(/reproducibility & execution context/i)).toBeInTheDocument();
    });

    // Unconfigured values should have distinct unconfigured labels
    expect(screen.getByText("Default (unconfigured)")).toBeInTheDocument();
    expect(screen.getByText("Unconfigured (unsupervised/image)")).toBeInTheDocument();
    expect(screen.getByText("Unconfigured")).toBeInTheDocument();

    // Missing fingerprint should show Unavailable
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
  });

  it("renders error alert on request failure", async () => {
    mockAxiosGet.mockRejectedValueOnce(new Error("Network connection lost"));

    render(<ReproducibilityDetails jobId="job-fail-999" />);

    await waitFor(() => {
      expect(screen.getByText(/Network connection lost/i)).toBeInTheDocument();
    });
  });

  it("copies snapshot json to clipboard when button clicked", async () => {
    const user = userEvent.setup();
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: writeTextMock,
      },
      writable: true,
      configurable: true,
    });

    mockAxiosGet.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          job_id: "job-copy-test",
          has_snapshot: true,
          reproducibility: SAMPLE_SNAPSHOT,
        },
      },
    });

    render(<ReproducibilityDetails jobId="job-copy-test" />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /export snapshot json/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /export snapshot json/i }));

    expect(writeTextMock).toHaveBeenCalledWith(JSON.stringify(SAMPLE_SNAPSHOT, null, 2));
    await waitFor(() => {
      expect(screen.getByText(/copied json/i)).toBeInTheDocument();
    });
  });
});
