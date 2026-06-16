/**
 * Tests for GenericLayerNode component.
 *
 * Phase 1 Week 2 - Registry-driven node component tests.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import GenericLayerNode from "../GenericLayerNode";
import * as registryHook from "../../../hooks/useLayerRegistry";
import type { LayerRegistryResponse, LayerSpec } from "../../../types/registry";

// Mock the registry hook
vi.mock("../../../hooks/useLayerRegistry");

// Helper to wrap component with ReactFlow provider
const renderWithProvider = (component: React.ReactElement) => {
  return render(<ReactFlowProvider>{component}</ReactFlowProvider>);
};

const mockRegistryData: LayerRegistryResponse = {
  categories: ["core", "recurrent", "utility"],
  layers: {
    core: [
      {
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
      },
    ],
    recurrent: [
      {
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
        },
      },
    ],
    utility: [
      {
        type_key: "concatenate",
        display_name: "Concatenate",
        category: "utility",
        keras_class: "tf.keras.layers.Concatenate",
        merge: true,
        description: "Concatenate layer",
        params: {
          axis: {
            name: "axis",
            type: "int" as const,
            required: false,
            default: -1,
            description: "Concatenation axis",
          },
        },
      },
    ],
  },
};

describe("GenericLayerNode", () => {
  beforeEach(() => {
    // Reset the cached registry in the module
    vi.mocked(registryHook.useLayerRegistry).mockReturnValue({
      registry: mockRegistryData,
      isLoading: false,
      error: null,
    });

    vi.mocked(registryHook.getLayerSpec).mockImplementation((typeKey: string) => {
      for (const layerSpecs of Object.values(mockRegistryData.layers)) {
        const spec = layerSpecs.find((s) => s.type_key === typeKey);
        if (spec) return spec;
      }
      return null;
    });
  });

  it("renders dense layer correctly", () => {
    const data = {
      layerType: "dense",
      params: { units: 64, activation: "relu" },
    };

    renderWithProvider(<GenericLayerNode data={data} selected={false} id="node-1" />);

    expect(screen.getByText("Dense")).toBeInTheDocument();
    expect(screen.getByText("DN")).toBeInTheDocument();
    expect(screen.getByText(/units: 64/)).toBeInTheDocument();
  });

  it("renders lstm layer correctly", () => {
    const data = {
      layerType: "lstm",
      params: { units: 128 },
    };

    renderWithProvider(<GenericLayerNode data={data} selected={false} id="node-2" />);

    expect(screen.getByText("LSTM")).toBeInTheDocument();
    expect(screen.getByText("LS")).toBeInTheDocument();
    expect(screen.getByText(/units: 128/)).toBeInTheDocument();
  });

  it("renders concatenate with two input handles", () => {
    const data = {
      layerType: "concatenate",
      params: { axis: -1 },
    };

    const { container } = renderWithProvider(<GenericLayerNode data={data} selected={false} id="node-3" />);

    expect(screen.getByText("Concatenate")).toBeInTheDocument();

    // Check for multiple input handles
    const inputHandles = container.querySelectorAll(".react-flow__handle-top");
    expect(inputHandles.length).toBeGreaterThanOrEqual(2);
  });

  it("shows skeleton when registry loading", () => {
    vi.mocked(registryHook.useLayerRegistry).mockReturnValue({
      registry: null,
      isLoading: true,
      error: null,
    });

    const data = {
      layerType: "dense",
      params: {},
    };

    const { container } = renderWithProvider(<GenericLayerNode data={data} selected={false} id="node-4" />);

    // Check for skeleton animation
    expect(container.querySelector(".animate-pulse")).toBeInTheDocument();
  });

  it("shows error for unknown layer type", () => {
    vi.mocked(registryHook.getLayerSpec).mockReturnValue(null);

    const data = {
      layerType: "unknown_xyz",
      params: {},
    };

    renderWithProvider(<GenericLayerNode data={data} selected={false} id="node-5" />);

    expect(screen.getByText(/Unknown layer/)).toBeInTheDocument();
    expect(screen.getByText(/unknown_xyz/)).toBeInTheDocument();
  });

  it("applies correct category color", () => {
    const data = {
      layerType: "dense",
      params: { units: 64 },
    };

    const { container } = renderWithProvider(<GenericLayerNode data={data} selected={false} id="node-6" />);

    // Check for blue color class (core category)
    expect(container.querySelector(".bg-blue-500")).toBeInTheDocument();
  });
});
