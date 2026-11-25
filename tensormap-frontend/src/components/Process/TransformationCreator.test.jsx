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

  it("updates selectedFeature when features prop loads asynchronously", async () => {
    const user = userEvent.setup();
    const handleAdd = vi.fn();
    const { rerender } = render(<TransformationCreator features={[]} onAdd={handleAdd} />);

    const button = screen.getByRole("button", { name: /add transformation/i });
    expect(button).toBeDisabled();

    rerender(<TransformationCreator features={sampleFeatures} onAdd={handleAdd} />);
    expect(button).not.toBeDisabled();

    await user.click(button);
    expect(handleAdd).toHaveBeenCalledWith({
      feature: "sepal_length",
      transformation: "One Hot Encoding",
    });
  });

  it("updates selectedFeature when features list changes to different columns", async () => {
    const user = userEvent.setup();
    const handleAdd = vi.fn();
    const { rerender } = render(
      <TransformationCreator features={sampleFeatures} onAdd={handleAdd} />,
    );

    rerender(<TransformationCreator features={["price", "bedrooms"]} onAdd={handleAdd} />);

    const button = screen.getByRole("button", { name: /add transformation/i });
    await user.click(button);

    expect(handleAdd).toHaveBeenCalledWith({
      feature: "price",
      transformation: "One Hot Encoding",
    });
  });

  it("handles Fill Missing Values with default auto strategy in parameters", async () => {
    const user = userEvent.setup();
    const handleAdd = vi.fn();
    render(<TransformationCreator features={sampleFeatures} onAdd={handleAdd} />);

    // Select Fill Missing Values from dropdown
    const triggers = screen.getAllByRole("combobox");
    const transformTrigger = triggers[1];
    await user.click(transformTrigger);

    const option = await screen.findByRole("option", { name: "Fill Missing Values" });
    await user.click(option);

    const button = screen.getByRole("button", { name: /add transformation/i });
    await user.click(button);

    expect(handleAdd).toHaveBeenCalledWith({
      feature: "sepal_length",
      transformation: "Fill Missing Values",
      params: { strategy: "auto" },
    });
  });
});
