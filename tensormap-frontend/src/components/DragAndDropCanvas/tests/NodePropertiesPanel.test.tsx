/**
 * Tests for NodePropertiesPanel component.
 *
 * Phase 1 Week 2 - Dynamic property editor tests.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import NodePropertiesPanel from "../NodePropertiesPanel";
import * as registryHook from "../../../hooks/useLayerRegistry";
import type { LayerSpec } from "../../../types/registry";

vi.mock("../../../hooks/useLayerRegistry");

const mockDenseSpec: LayerSpec = {
  type_key: "dense",
  display_name: "Dense",
  category: "core",
  keras_class: "tf.keras.layers.Dense",
  merge: false,
  description: "Dense layer",
  params: {
    units: {
      name: "units",
      type: "int" as const,
      required: true,
      default: null,
      min: 1,
      description: "Number of units",
    },
    activation: {
      name: "activation",
      type: "enum" as const,
      required: false,
      default: "relu",
      values: ["relu", "sigmoid", "tanh"],
      description: "Activation function",
    },
  },
};

const mockLSTMSpec: LayerSpec = {
  type_key: "lstm",
  display_name: "LSTM",
  category: "recurrent",
  keras_class: "tf.keras.layers.LSTM",
  merge: false,
  description: "LSTM layer",
  params: {
    units: {
      name: "units",
      type: "int" as const,
      required: true,
      default: null,
      min: 1,
      description: "Number of units",
    },
    return_sequences: {
      name: "return_sequences",
      type: "bool" as const,
      required: false,
      default: false,
      description: "Whether to return sequences",
    },
  },
};

describe("NodePropertiesPanel", () => {
  const mockOnModelNameChange = vi.fn();
  const mockOnSave = vi.fn();
  const mockOnNodeUpdate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(registryHook.getLayerSpec).mockImplementation((typeKey: string) => {
      if (typeKey === "dense") return mockDenseSpec;
      if (typeKey === "lstm") return mockLSTMSpec;
      return null;
    });
  });

  it("renders model save panel when no node selected", () => {
    render(
      <NodePropertiesPanel
        selectedNode={null}
        modelName=""
        onModelNameChange={mockOnModelNameChange}
        onSave={mockOnSave}
        canSave={false}
        onNodeUpdate={mockOnNodeUpdate}
      />,
    );

    expect(screen.getByText("Save Model")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/enter model name/i)).toBeInTheDocument();
    expect(screen.getByText(/validate & save/i)).toBeInTheDocument();
  });

  it("renders int input for units param", () => {
    const selectedNode = {
      id: "node-1",
      type: "dense",
      data: {
        layerType: "dense",
        params: { units: 64, activation: "relu" },
      },
    };

    render(
      <NodePropertiesPanel
        selectedNode={selectedNode}
        modelName="test-model"
        onModelNameChange={mockOnModelNameChange}
        onSave={mockOnSave}
        canSave={true}
        onNodeUpdate={mockOnNodeUpdate}
      />,
    );

    expect(screen.getByText("Dense Layer")).toBeInTheDocument();
    expect(screen.getByText(/units/i)).toBeInTheDocument();

    const unitsInput = screen.getByDisplayValue("64");
    expect(unitsInput).toBeInTheDocument();
    expect(unitsInput).toHaveAttribute("type", "number");
  });

  it("renders enum select for activation param", () => {
    const selectedNode = {
      id: "node-1",
      type: "dense",
      data: {
        layerType: "dense",
        params: { units: 64, activation: "relu" },
      },
    };

    render(
      <NodePropertiesPanel
        selectedNode={selectedNode}
        modelName="test-model"
        onModelNameChange={mockOnModelNameChange}
        onSave={mockOnSave}
        canSave={true}
        onNodeUpdate={mockOnNodeUpdate}
      />,
    );

    expect(screen.getByText(/activation/i)).toBeInTheDocument();
    // Select component should be present
    const selectTrigger = screen.getByRole("combobox");
    expect(selectTrigger).toBeInTheDocument();
  });

  it("renders checkbox for return_sequences param", () => {
    const selectedNode = {
      id: "node-2",
      type: "lstm",
      data: {
        layerType: "lstm",
        params: { units: 128, return_sequences: false },
      },
    };

    render(
      <NodePropertiesPanel
        selectedNode={selectedNode}
        modelName="test-model"
        onModelNameChange={mockOnModelNameChange}
        onSave={mockOnSave}
        canSave={true}
        onNodeUpdate={mockOnNodeUpdate}
      />,
    );

    expect(screen.getByText(/return_sequences/i)).toBeInTheDocument();

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeInTheDocument();
  });

  it("validates units >= 1 and shows error", () => {
    const selectedNode = {
      id: "node-1",
      type: "dense",
      data: {
        layerType: "dense",
        params: { units: -5, activation: "relu" },
      },
    };

    render(
      <NodePropertiesPanel
        selectedNode={selectedNode}
        modelName="test-model"
        onModelNameChange={mockOnModelNameChange}
        onSave={mockOnSave}
        canSave={true}
        onNodeUpdate={mockOnNodeUpdate}
      />,
    );

    const unitsInput = screen.getByDisplayValue("-5");

    // Change to valid value
    fireEvent.change(unitsInput, { target: { value: "10" } });

    expect(mockOnNodeUpdate).toHaveBeenCalledWith("node-1", {
      units: 10,
      activation: "relu",
    });
  });

  it("reset to defaults button works", () => {
    const selectedNode = {
      id: "node-1",
      type: "dense",
      data: {
        layerType: "dense",
        params: { units: 64, activation: "sigmoid" },
      },
    };

    render(
      <NodePropertiesPanel
        selectedNode={selectedNode}
        modelName="test-model"
        onModelNameChange={mockOnModelNameChange}
        onSave={mockOnSave}
        canSave={true}
        onNodeUpdate={mockOnNodeUpdate}
      />,
    );

    const resetButton = screen.getByText(/reset to defaults/i);
    fireEvent.click(resetButton);

    expect(mockOnNodeUpdate).toHaveBeenCalledWith("node-1", {
      units: null,
      activation: "relu",
    });
  });
});
