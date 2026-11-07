import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TransformationCreator from "./TransformationCreator";

describe("TransformationCreator", () => {
  const sampleFeatures = ["sepal_length", "sepal_width", "petal_length"];

  it("renders feature and transformation dropdowns and disabled add button when unselected", () => {
    const handleAdd = vi.fn();
    render(<TransformationCreator features={[]} onAdd={handleAdd} />);
    const button = screen.getByRole("button", { name: /add transformation/i });
    expect(button).toBeDisabled();
  });

  it("renders with default selections when features are provided", () => {
    const handleAdd = vi.fn();
    render(<TransformationCreator features={sampleFeatures} onAdd={handleAdd} />);
    const button = screen.getByRole("button", { name: /add transformation/i });
    expect(button).not.toBeDisabled();
  });

  it("calls onAdd with selected feature and transformation when clicked", async () => {
    const user = userEvent.setup();
    const handleAdd = vi.fn();
    render(<TransformationCreator features={sampleFeatures} onAdd={handleAdd} />);

    const button = screen.getByRole("button", { name: /add transformation/i });
    await user.click(button);

    expect(handleAdd).toHaveBeenCalledTimes(1);
    expect(handleAdd).toHaveBeenCalledWith({
      feature: "sepal_length",
      transformation: "One Hot Encoding",
    });
  });
});
