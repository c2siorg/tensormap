/**
 * Tests for Helpers module.
 *
 * Phase 1 Week 2 - Registry-driven validation tests.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { canSaveModel, generateModelJSON } from "../Helpers";
import * as registryHook from "../../../hooks/useLayerRegistry";
import type { LayerRegistryResponse, LayerSpec } from "../../../types/registry";

vi.mock("../../../hooks/useLayerRegistry");

const mockRegistryData: LayerRegistryResponse = {
  categories: ["core", "utility"],
  layers: {
    core: [
      {
        type_key: "input",
        display_name: "Input",
        category: "core",
        keras_class: "tf.keras.layers.InputLayer",
        merge: false,
        description: "Input layer",
        params: {
          shape: {
            name: "shape",
            type: "int" as const,
            required: true,
            default: null,
            min: 1,
            description: "Input shape",
          },
        },
      },
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

describe("Helpers", () => {
  beforeEach(() => {
    vi.mocked(registryHook.getLayerSpec).mockImplementation((typeKey: string) => {
      for (const layerSpecs of Object.values(mockRegistryData.layers)) {
        const spec = layerSpecs.find((s) => s.type_key === typeKey);
        if (spec) return spec;
      }
      return null;
    });

    vi.mocked(registryHook.getAllLayerSpecs).mockReturnValue(
      Object.values(mockRegistryData.layers).flat(),
    );
  });

  describe("canSaveModel", () => {
    it("returns valid for correct Dense network", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            data: {
              layerType: "input",
              params: { shape: 784 },
            },
          },
          {
            id: "node-2",
            type: "genericlayer",
            data: {
              layerType: "dense",
              params: { units: 128, activation: "relu" },
            },
          },
        ],
        edges: [{ source: "node-1", target: "node-2" }],
      };

      const result = canSaveModel("test-model", modelData, mockRegistryData);

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it("catches missing required units", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            data: {
              layerType: "dense",
              params: { units: null, activation: "relu" },
            },
          },
        ],
        edges: [],
      };

      const result = canSaveModel("test-model", modelData, mockRegistryData);

      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(1); // units required (single node is considered connected)
      expect(result.errors[0].field).toBe("units");
      expect(result.errors[0].message).toContain("required");
    });

    it("catches Concatenate with 1 edge", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            data: {
              layerType: "input",
              params: { shape: 784 },
            },
          },
          {
            id: "node-2",
            type: "genericlayer",
            data: {
              layerType: "concatenate",
              params: { axis: -1 },
            },
          },
        ],
        edges: [{ source: "node-1", target: "node-2" }],
      };

      const result = canSaveModel("test-model", modelData, mockRegistryData);

      expect(result.valid).toBe(false);
      const concatError = result.errors.find((e) => e.field === "connections");
      expect(concatError).toBeDefined();
      expect(concatError?.message).toContain("at least 2 inputs");
    });

    it("catches negative units value", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            data: {
              layerType: "dense",
              params: { units: -5, activation: "relu" },
            },
          },
        ],
        edges: [],
      };

      const result = canSaveModel("test-model", modelData, mockRegistryData);

      expect(result.valid).toBe(false);
      const unitsError = result.errors.find((e) => e.field === "units");
      expect(unitsError).toBeDefined();
      expect(unitsError?.message).toContain(">=");
    });

    it("catches empty model name", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            data: {
              layerType: "dense",
              params: { units: 64, activation: "relu" },
            },
          },
        ],
        edges: [],
      };

      const result = canSaveModel("", modelData, mockRegistryData);

      expect(result.valid).toBe(false);
      const nameError = result.errors.find((e) => e.field === "name");
      expect(nameError).toBeDefined();
    });

    it("catches disconnected graph", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            data: {
              layerType: "input",
              params: { shape: 784 },
            },
          },
          {
            id: "node-2",
            type: "genericlayer",
            data: {
              layerType: "dense",
              params: { units: 128, activation: "relu" },
            },
          },
        ],
        edges: [], // No edges - disconnected
      };

      const result = canSaveModel("test-model", modelData, mockRegistryData);

      expect(result.valid).toBe(false);
      const connectivityError = result.errors.find((e) => e.field === "connectivity");
      expect(connectivityError).toBeDefined();
    });
  });

  describe("generateModelJSON", () => {
    it("strips visual properties", () => {
      const modelData = {
        nodes: [
          {
            id: "node-1",
            type: "genericlayer",
            position: { x: 100, y: 200 },
            selected: true,
            data: {
              layerType: "dense",
              params: { units: 64, activation: "relu" },
              label: "Dense",
            },
          },
        ],
        edges: [
          {
            id: "edge-1",
            source: "node-1",
            target: "node-2",
            animated: true,
            style: { stroke: "red" },
          },
        ],
      };

      const result = generateModelJSON(modelData);

      expect(result.nodes[0]).toEqual({
        id: "node-1",
        type: "genericlayer",
        position: { x: 100, y: 200 },
        data: {
          params: { units: 64, activation: "relu" },
        },
      });

      expect(result.edges[0]).toEqual({
        source: "node-1",
        target: "node-2",
      });
    });
  });
});
