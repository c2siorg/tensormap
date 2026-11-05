/**
 * Tests for FairnessAuditPanel component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import FairnessAuditPanel from "../FairnessAuditPanel";
import axios from "@/shared/Axios";

vi.mock("@/shared/Axios");

describe("FairnessAuditPanel", () => {
  const mockFairnessData = {
    job_id: "test-job-789",
    sensitive_feature: "gender",
    privileged_group: "male",
    group_metrics: {
      male: {
        group: "male",
        sample_count: 50,
        selection_rate: 0.8,
        accuracy: 0.9,
        precision: 0.85,
        recall: 0.88,
        f1_score: 0.86,
        true_positive_rate: 0.88,
        false_positive_rate: 0.12,
      },
      female: {
        group: "female",
        sample_count: 50,
        selection_rate: 0.72,
        accuracy: 0.88,
        precision: 0.82,
        recall: 0.84,
        f1_score: 0.83,
        true_positive_rate: 0.84,
        false_positive_rate: 0.1,
      },
    },
    disparities: {
      female: {
        group: "female",
        compared_to: "male",
        disparate_impact_ratio: 0.9,
        demographic_parity_difference: -0.08,
        equal_opportunity_difference: -0.04,
        average_odds_difference: -0.03,
        passes_80_percent_rule: true,
      },
    },
    overall_summary: {
      compliant_with_80_percent_rule: true,
      groups_evaluated: 2,
      total_samples: 100,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders prompt initially before auditing", () => {
    render(<FairnessAuditPanel jobId="test-job-789" features={["gender", "race"]} />);

    expect(
      screen.getByText("Algorithmic Fairness & Demographic Parity Audit")
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Select or specify a sensitive feature above and click "Audit Fairness"/i)
    ).toBeInTheDocument();
  });

  it("pre-populates first feature and renders feature chips", () => {
    render(<FairnessAuditPanel jobId="test-job-789" features={["gender", "race"]} />);

    const input = screen.getByRole("textbox") as HTMLInputElement;
    expect(input.value).toBe("gender");
    expect(screen.getByText("race")).toBeInTheDocument();
  });

  it("triggers API call and displays fairness audit results on submit", async () => {
    (axios.get as any).mockResolvedValueOnce({
      data: mockFairnessData,
    });

    render(<FairnessAuditPanel jobId="test-job-789" features={["gender"]} />);

    const auditButton = screen.getByRole("button", { name: /Audit Fairness/i });
    fireEvent.click(auditButton);

    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith(
        expect.stringContaining("/fairness"),
        expect.objectContaining({
          params: { sensitive_feature: "gender" },
        })
      );
      expect(screen.getByText("Compliant (Pass)")).toBeInTheDocument();
      expect(screen.getAllByText("female").length).toBeGreaterThan(0);
      expect(screen.getByText("0.9000")).toBeInTheDocument();
      expect(screen.getByText("80.0%")).toBeInTheDocument();
      expect(screen.getByText("72.0%")).toBeInTheDocument();
    });
  });

  it("handles disparity violation (< 80% rule)", async () => {
    const violatingData = {
      ...mockFairnessData,
      disparities: {
        female: {
          ...mockFairnessData.disparities.female,
          disparate_impact_ratio: 0.65,
          passes_80_percent_rule: false,
        },
      },
      overall_summary: {
        ...mockFairnessData.overall_summary,
        compliant_with_80_percent_rule: false,
      },
    };

    (axios.get as any).mockResolvedValueOnce({
      data: violatingData,
    });

    render(<FairnessAuditPanel jobId="test-job-789" features={["gender"]} />);

    fireEvent.click(screen.getByRole("button", { name: /Audit Fairness/i }));

    await waitFor(() => {
      expect(screen.getByText("Disparity Detected")).toBeInTheDocument();
      expect(screen.getByText("Flagged (< 0.8)")).toBeInTheDocument();
    });
  });

  it("displays error message when API rejects the request", async () => {
    (axios.get as any).mockRejectedValueOnce({
      response: { data: { detail: "Feature 'unknown_attr' not found in dataset" } },
    });

    render(<FairnessAuditPanel jobId="test-job-789" />);

    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "unknown_attr" } });
    fireEvent.click(screen.getByRole("button", { name: /Audit Fairness/i }));

    await waitFor(() => {
      expect(screen.getByText(/Audit Error:/i)).toBeInTheDocument();
      expect(screen.getByText(/Feature 'unknown_attr' not found in dataset/i)).toBeInTheDocument();
    });
  });
});
