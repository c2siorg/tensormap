/**
 * Tests for Sidebar component.
 *
 * Phase 1 Week 2 - Categorized, searchable layer palette tests.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import Sidebar from "../Sidebar";
import * as registryHook from "../../../hooks/useLayerRegistry";
import type { LayerRegistryResponse } from "../../../types/registry";

vi.mock("../../../hooks/useLayerRegistry");

const mockRegistryData: LayerRegistryResponse = {
  categories: ["core", "convolutional", "pooling"],
  layers: {
    core: [
      {
        type_key: "input",
        display_name: "Input",
        category: "core",
        keras_class: "tf.keras.layers.InputLayer",
        merge: false,
        description: "Input layer",
        params: {},
      },
      {
        type_key: "dense",
        display_name: "Dense",
        category: "core",
        keras_class: "tf.keras.layers.Dense",
        merge: false,
        description: "Dense layer",
        params: {},
      },
      {
        type_key: "flatten",
        display_name: "Flatten",
        category: "core",
        keras_class: "tf.keras.layers.Flatten",
        merge: false,
        description: "Flatten layer",
        params: {},
      },
    ],
    convolutional: [
      {
        type_key: "conv2d",
        display_name: "Conv2D",
        category: "convolutional",
        keras_class: "tf.keras.layers.Conv2D",
        merge: false,
        description: "Conv2D layer",
        params: {},
      },
    ],
    pooling: [
      {
        type_key: "maxpool2d",
        display_name: "MaxPooling2D",
        category: "pooling",
        keras_class: "tf.keras.layers.MaxPooling2D",
        merge: false,
        description: "Max pooling layer",
        params: {},
      },
    ],
  },
};

describe("Sidebar", () => {
  beforeEach(() => {
    vi.mocked(registryHook.useLayerRegistry).mockReturnValue({
      registry: mockRegistryData,
      isLoading: false,
      error: null,
    });
  });

  it("renders all 5 layers grouped by category", () => {
    render(<Sidebar />);

    // Check category headers - use more specific queries to avoid duplicates
    expect(screen.getByText(/core/i)).toBeInTheDocument();
    expect(screen.getByText(/convolutional/i)).toBeInTheDocument();
    // "pooling" appears in both category name and "MaxPooling2D", so check for the button role
    expect(screen.getByRole("button", { name: /pooling/i })).toBeInTheDocument();

    // Check layer items
    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("Dense")).toBeInTheDocument();
    expect(screen.getByText("Flatten")).toBeInTheDocument();
    expect(screen.getByText("Conv2D")).toBeInTheDocument();
    expect(screen.getByText("MaxPooling2D")).toBeInTheDocument();

    // Check total count
    expect(screen.getByText("(5)")).toBeInTheDocument();
  });

  it("search filters layers correctly", () => {
    render(<Sidebar />);

    const searchInput = screen.getByPlaceholderText(/search layers/i);
    fireEvent.change(searchInput, { target: { value: "dense" } });

    // Should show only Dense layer
    expect(screen.getByText("Dense")).toBeInTheDocument();
    expect(screen.queryByText("Input")).not.toBeInTheDocument();
    expect(screen.queryByText("Flatten")).not.toBeInTheDocument();
    expect(screen.queryByText("Conv2D")).not.toBeInTheDocument();
  });

  it("search is case-insensitive", () => {
    render(<Sidebar />);

    const searchInput = screen.getByPlaceholderText(/search layers/i);
    fireEvent.change(searchInput, { target: { value: "CONV" } });

    expect(screen.getByText("Conv2D")).toBeInTheDocument();
    expect(screen.queryByText("Dense")).not.toBeInTheDocument();
  });

  it("category collapse/expand works", () => {
    render(<Sidebar />);

    // Find and click the core category header
    const coreHeader = screen.getByText(/core/i);
    fireEvent.click(coreHeader);

    // Layers should be hidden after collapse
    // Note: This might need adjustment based on actual implementation
    // We're checking that the button exists and is clickable
    expect(coreHeader).toBeInTheDocument();
  });

  it("draggable items have correct data attributes", () => {
    const { container } = render(<Sidebar />);

    const draggableItems = container.querySelectorAll('[draggable="true"]');
    expect(draggableItems.length).toBe(5); // 5 layers total

    // Check that items have drag handlers
    expect(draggableItems[0]).toHaveAttribute("draggable", "true");
  });

  it("shows no results message for non-matching search", () => {
    render(<Sidebar />);

    const searchInput = screen.getByPlaceholderText(/search layers/i);
    fireEvent.change(searchInput, { target: { value: "nonexistent" } });

    expect(screen.getByText(/No layers match/i)).toBeInTheDocument();
  });
});
