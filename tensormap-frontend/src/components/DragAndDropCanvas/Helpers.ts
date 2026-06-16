/**
 * Helper functions for model validation and JSON generation.
 *
 * Phase 1 Week 2 - Fully registry-driven validation.
 */

import { getLayerSpec, getAllLayerSpecs } from "../../hooks/useLayerRegistry";
import type { LayerSpec, ParamSpec, LayerRegistryResponse } from "../../types/registry";
import { LEGACY_TYPE_MAP } from "../../types/registry";

interface ModelData {
  nodes: any[];
  edges: any[];
}

interface ValidationError {
  nodeId: string;
  field: string;
  message: string;
}

/**
 * Get the registry type key from a node type (handles legacy types).
 */
function getRegistryTypeKey(nodeType: string): string {
  return LEGACY_TYPE_MAP[nodeType] || nodeType;
}

/**
 * Validate a single parameter value against its spec.
 */
function validateParamValue(paramName: string, value: any, spec: ParamSpec): string | null {
  // Check required
  if (spec.required && (value === null || value === undefined || value === "")) {
    return `${paramName} is required`;
  }

  // Skip further validation if empty and not required
  if (value === null || value === undefined || value === "") {
    return null;
  }

  // Type-specific validation
  if (spec.type === "int") {
    const num = Number(value);
    if (!Number.isInteger(num)) {
      return `${paramName} must be an integer`;
    }
    if (spec.min !== undefined && spec.min !== null && num < spec.min) {
      return `${paramName} must be >= ${spec.min}`;
    }
    if (spec.max !== undefined && spec.max !== null && num > spec.max) {
      return `${paramName} must be <= ${spec.max}`;
    }
  }

  if (spec.type === "float") {
    const num = Number(value);
    if (isNaN(num)) {
      return `${paramName} must be a number`;
    }
    if (spec.min !== undefined && spec.min !== null && num < spec.min) {
      return `${paramName} must be >= ${spec.min}`;
    }
    if (spec.max !== undefined && spec.max !== null && num > spec.max) {
      return `${paramName} must be <= ${spec.max}`;
    }
  }

  return null;
}

/**
 * Validate all parameters for a node.
 */
function validateNodeParams(node: any, layerSpec: LayerSpec): ValidationError[] {
  const errors: ValidationError[] = [];
  const params = node.data?.params || {};

  // Check each parameter in the spec
  for (const [paramName, paramSpec] of Object.entries(layerSpec.params)) {
    const value = params[paramName];
    const error = validateParamValue(paramName, value, paramSpec);
    if (error) {
      errors.push({
        nodeId: node.id,
        field: paramName,
        message: error,
      });
    }
  }

  return errors;
}

/**
 * Check if the graph is fully connected (no isolated nodes).
 */
function isGraphConnected(graph: ModelData): boolean {
  if (!graph.nodes || graph.nodes.length === 0) return false;
  if (graph.nodes.length === 1) return true; // Single node is considered connected

  const visited = new Set<string>();
  const firstNodeId = graph.nodes[0].id;
  const queue = [firstNodeId];
  visited.add(firstNodeId);

  while (queue.length > 0) {
    const currentNodeId = queue.shift()!;
    const adjacentNodes = graph.edges
      .filter((edge: any) => edge.source === currentNodeId || edge.target === currentNodeId)
      .map((edge: any) => (edge.source === currentNodeId ? edge.target : edge.source));

    for (const nodeId of adjacentNodes) {
      if (!visited.has(nodeId)) {
        queue.push(nodeId);
        visited.add(nodeId);
      }
    }
  }

  return visited.size === graph.nodes.length;
}

/**
 * Validate that merge layers (like Concatenate) have at least 2 incoming edges.
 */
function validateMergeLayerEdges(
  node: any,
  layerSpec: LayerSpec,
  edges: any[],
): ValidationError | null {
  if (!layerSpec.merge) return null;

  const incomingEdges = edges.filter((edge) => edge.target === node.id);
  if (incomingEdges.length < 2) {
    return {
      nodeId: node.id,
      field: "connections",
      message: `${layerSpec.display_name} requires at least 2 inputs, found ${incomingEdges.length}`,
    };
  }

  return null;
}

/**
 * Registry-driven validation function.
 * Returns { valid: boolean, errors: ValidationError[] }
 */
export function canSaveModel(
  modelName: string,
  modelData: ModelData,
  registry?: LayerRegistryResponse | null,
): { valid: boolean; errors: ValidationError[] } {
  const errors: ValidationError[] = [];

  // Check model name
  if (!modelName || modelName.trim() === "") {
    errors.push({
      nodeId: "model",
      field: "name",
      message: "Model name is required",
    });
    return { valid: false, errors };
  }

  // Check nodes exist
  if (!modelData.nodes || modelData.nodes.length === 0) {
    errors.push({
      nodeId: "model",
      field: "nodes",
      message: "Model must have at least one layer",
    });
    return { valid: false, errors };
  }

  // If registry not loaded yet, we can't validate params
  // This shouldn't happen in practice since registry loads on mount
  if (!registry) {
    // For backward compatibility, just check basic connectivity
    if (!isGraphConnected(modelData)) {
      errors.push({
        nodeId: "model",
        field: "connectivity",
        message: "All layers must be connected",
      });
    }
    return { valid: errors.length === 0, errors };
  }

  // Validate each node using registry
  for (const node of modelData.nodes) {
    const nodeType = node.data?.layerType || node.type;
    const registryType = getRegistryTypeKey(nodeType);
    const layerSpec = getLayerSpec(registryType);

    if (!layerSpec) {
      errors.push({
        nodeId: node.id,
        field: "type",
        message: `Unknown layer type: ${nodeType}`,
      });
      continue;
    }

    // Validate parameters
    const paramErrors = validateNodeParams(node, layerSpec);
    errors.push(...paramErrors);

    // Validate merge layer connections
    const mergeError = validateMergeLayerEdges(node, layerSpec, modelData.edges);
    if (mergeError) {
      errors.push(mergeError);
    }
  }

  // Check graph connectivity
  if (!isGraphConnected(modelData)) {
    errors.push({
      nodeId: "model",
      field: "connectivity",
      message: "All layers must be connected",
    });
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Simplified boolean version for backward compatibility.
 * Uses the cached registry from useLayerRegistry for validation.
 */
export function canSaveModelSimple(modelName: string, modelData: ModelData): boolean {
  // Import the cached registry from the module-level cache
  const registry = getAllLayerSpecs().length > 0 ? buildRegistryFromSpecs() : null;
  const result = canSaveModel(modelName, modelData, registry);
  return result.valid;
}

/**
 * Helper to build LayerRegistryResponse from cached specs.
 */
function buildRegistryFromSpecs(): LayerRegistryResponse | null {
  const allSpecs = getAllLayerSpecs();
  if (allSpecs.length === 0) return null;

  const layersByCategory: Record<string, LayerSpec[]> = {};
  const categoriesSet = new Set<string>();

  for (const spec of allSpecs) {
    if (!layersByCategory[spec.category]) {
      layersByCategory[spec.category] = [];
    }
    layersByCategory[spec.category].push(spec);
    categoriesSet.add(spec.category);
  }

  return {
    categories: Array.from(categoriesSet).sort(),
    layers: layersByCategory,
  };
}

/**
 * Strips visual-only properties from a ReactFlow graph snapshot using
 * immutable operations (no destructive delete mutations).
 */
export function generateModelJSON(modelData: ModelData) {
  const nodes = modelData.nodes.map((node) => ({
    id: node.id,
    type: node.type,
    position: node.position,
    data: { params: node.data.params },
  }));

  const edges = modelData.edges.map((edge) => ({
    source: edge.source,
    target: edge.target,
  }));

  return { nodes, edges };
}
