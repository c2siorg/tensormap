import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DatasetQualityReport from "../DatasetQualityReport";

describe("DatasetQualityReport", () => {
  it("shows an accessible loading state", () => {
    render(<DatasetQualityReport loading />);
    expect(screen.getByRole("status")).toHaveTextContent("Analyzing the selected dataset");
  });

  it("shows a clean report", () => {
    render(
      <DatasetQualityReport
        report={{
          status: "clean",
          summary: { rows: 4, columns: 2 },
          issues: [],
          limitations: ["Duplicate CSV headers may be normalized by pandas during parsing."],
        }}
      />,
    );
    expect(screen.getByText("Dataset quality looks good")).toBeInTheDocument();
    expect(screen.getByText(/Checked 4 rows across 2 columns/)).toBeInTheDocument();
    expect(screen.getByText(/Duplicate CSV headers may be normalized/)).toBeInTheDocument();
  });

  it("distinguishes blocking findings from warnings and shows affected-column details", () => {
    render(
      <DatasetQualityReport
        report={{
          status: "blocked",
          summary: { rows: 3, columns: 2 },
          issues: [
            {
              code: "target_not_found",
              severity: "error",
              blocking: true,
              columns: ["label"],
              message: "Target column 'label' was not found.",
            },
            {
              code: "missing_values",
              severity: "warning",
              blocking: false,
              columns: ["feature"],
              count: 1,
              percentage: 33.33,
              message: "One feature value is missing.",
            },
          ],
          limitations: [],
        }}
      />,
    );
    expect(screen.getByText("Resolve blocking dataset issues before training")).toBeInTheDocument();
    expect(screen.getByText("Blocking error")).toBeInTheDocument();
    expect(screen.getByText("Warning")).toBeInTheDocument();
    expect(screen.getByText("Target column 'label' was not found.")).toBeInTheDocument();
    expect(screen.getByText("Affected columns: feature")).toBeInTheDocument();
    expect(screen.getByText("Missing values: 1 (33.33% of rows)")).toBeInTheDocument();
  });

  it("shows analysis failures without hiding existing training validation", () => {
    render(<DatasetQualityReport error="Please try again." />);
    expect(screen.getByText("Dataset quality could not be analyzed")).toBeInTheDocument();
    expect(screen.getByText(/Existing training validation still applies/)).toBeInTheDocument();
  });
});
