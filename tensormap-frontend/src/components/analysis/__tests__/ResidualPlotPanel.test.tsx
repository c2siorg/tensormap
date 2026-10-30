/**
 * Unit tests for ResidualPlotPanel component.
 * @module
 */

import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import ResidualPlotPanel from "../ResidualPlotPanel";

describe("ResidualPlotPanel", () => {
  const defaultProps = {
    yPred: [1.2, 3.4, 5.6],
    yTrue: [1.0, 3.5, 5.5],
    residuals: [0.2, -0.1, 0.1],
    mae: 0.13333,
    mse: 0.02,
    rmse: 0.14142,
    r2: 0.98567,
  };

  it("renders panel title and all metric badges including RMSE and R²", () => {
    render(<ResidualPlotPanel {...defaultProps} />);

    expect(screen.getByText("Predicted vs Actual")).toBeInTheDocument();
    expect(screen.getByText("MAE: 0.1333")).toBeInTheDocument();
    expect(screen.getByText("MSE: 0.0200")).toBeInTheDocument();
    expect(screen.getByText("RMSE: 0.1414")).toBeInTheDocument();
    expect(screen.getByText("R²: 0.9857")).toBeInTheDocument();
  });

  it("omits R² badge when r2 is null", () => {
    render(<ResidualPlotPanel {...defaultProps} r2={null} />);

    expect(screen.getByText("MAE: 0.1333")).toBeInTheDocument();
    expect(screen.getByText("MSE: 0.0200")).toBeInTheDocument();
    expect(screen.getByText("RMSE: 0.1414")).toBeInTheDocument();
    expect(screen.queryByText(/R²:/)).not.toBeInTheDocument();
  });

  it("renders explanatory guide notes", () => {
    render(<ResidualPlotPanel {...defaultProps} />);

    expect(screen.getByText(/Red dashed line:/)).toBeInTheDocument();
    expect(screen.getByText(/Blue dots:/)).toBeInTheDocument();
  });
});
