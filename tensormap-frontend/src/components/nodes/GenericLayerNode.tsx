/**
 * GenericLayerNode - Universal node component for all layer types.
 *
 * Phase 1 Week 2 — Fully integrated with registry data from GET /layers endpoint
 *
 * This component replaces the need for individual node components (DenseNode, ConvNode, etc.)
 * by rendering any layer type based on its registry specification. The registry-driven approach
 * eliminates the need to create new React components when adding new layer types.
 *
 * Features:
 * - Category-based color coding for visual organization
 * - Two-letter abbreviation badges for quick layer identification
 * - Dynamic parameter summary display
 * - Support for merge layers (Concatenate) with multiple input handles
 * - ReactFlow integration with standard Handle components
 * - Skeleton loading state while registry loads
 * - Error state for unknown layer types
 */

import { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { useLayerRegistry, getLayerSpec } from "../../hooks/useLayerRegistry";
import {
  CATEGORY_BG_COLORS,
  CATEGORY_BORDER_COLORS,
  TYPE_ABBREVIATIONS,
} from "../../types/registry";

interface GenericLayerNodeData {
  layerType: string;
  label?: string;
  params: Record<string, any>;
}

/**
 * Get Tailwind background color class based on layer category.
 */
const getCategoryColor = (category: string): string => {
  return CATEGORY_BG_COLORS[category] || "bg-gray-500";
};

/**
 * Get Tailwind border color class based on layer category.
 */
const getBorderColor = (category: string): string => {
  return CATEGORY_BORDER_COLORS[category] || "border-gray-600";
};

/**
 * Get two-letter abbreviation for the layer type.
 */
const getLayerAbbreviation = (typeKey: string): string => {
  return TYPE_ABBREVIATIONS[typeKey] || typeKey.substring(0, 2).toUpperCase();
};

/**
 * Format parameter summary for display.
 * Shows the most important parameters in a compact format.
 */
const formatParamSummary = (params: Record<string, any>): string => {
  if (!params || Object.keys(params).length === 0) {
    return "No parameters";
  }

  const entries = Object.entries(params)
    .filter(([_key, value]) => value !== null && value !== undefined && value !== "")
    .slice(0, 2); // Show max 2 params

  if (entries.length === 0) return "Not configured";

  return entries
    .map(([key, value]) => {
      // Format value based on type
      if (typeof value === "boolean") return `${key}: ${value ? "yes" : "no"}`;
      if (typeof value === "number") return `${key}: ${value}`;
      if (typeof value === "string") return `${key}: ${value}`;
      return `${key}: ${JSON.stringify(value)}`;
    })
    .join(", ");
};

/**
 * Skeleton loader component shown while registry is loading.
 */
const SkeletonLoader = () => (
  <div className="rounded-lg border-2 border-gray-300 bg-white shadow-md min-w-[160px]">
    <div className="bg-gray-300 animate-pulse rounded-t-md h-10" />
    <div className="px-3 py-2">
      <div className="bg-gray-200 animate-pulse h-4 rounded" />
    </div>
  </div>
);

/**
 * Error state component shown for unknown layer types.
 */
const ErrorState = ({ layerType }: { layerType: string }) => (
  <div className="rounded-lg border-2 border-red-500 bg-white shadow-md min-w-[160px]">
    <div className="bg-red-500 rounded-t-md px-3 py-2 text-white">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Unknown</span>
        <span className="rounded bg-white/20 px-2 py-0.5 text-xs font-bold">??</span>
      </div>
    </div>
    <div className="px-3 py-2">
      <div className="text-xs text-red-600">Unknown layer: {layerType}</div>
    </div>
    <Handle
      type="target"
      position={Position.Top}
      style={{ background: "#555" }}
      className="h-3 w-3 rounded-full border-2 border-white"
    />
    <Handle
      type="source"
      position={Position.Bottom}
      style={{ background: "#555" }}
      className="h-3 w-3 rounded-full border-2 border-white"
    />
  </div>
);

/**
 * GenericLayerNode component.
 *
 * Renders a neural network layer node with:
 * - Colored header based on category
 * - Layer type abbreviation badge
 * - Display name
 * - Parameter summary
 * - Input/output handles for connections
 * - Multiple input handles for merge layers (Concatenate)
 */
const GenericLayerNode = memo(({ data, selected }: NodeProps<GenericLayerNodeData>) => {
  const { registry, isLoading } = useLayerRegistry();
  const { layerType, params } = data;

  // Show skeleton while loading
  if (isLoading || !registry) {
    return <SkeletonLoader />;
  }

  // Get layer spec from registry
  const layerSpec = getLayerSpec(layerType);

  // Show error state if layer type is unknown
  if (!layerSpec) {
    return <ErrorState layerType={layerType} />;
  }

  const category = layerSpec.category;
  const displayName = layerSpec.display_name;
  const isMergeLayer = layerSpec.merge;

  const categoryColor = getCategoryColor(category);
  const borderColor = getBorderColor(category);
  const abbreviation = getLayerAbbreviation(layerType);
  const paramSummary = formatParamSummary(params);

  return (
    <div
      className={`
        rounded-lg border-2 bg-white shadow-md transition-all
        ${selected ? `ring-2 ring-offset-1 ring-white ${borderColor}` : "border-gray-300"}
        min-w-[160px]
      `}
    >
      {/* Header with category color */}
      <div className={`${categoryColor} rounded-t-md px-3 py-2 text-white`}>
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-bold truncate">{displayName}</span>
          <span className="rounded-full bg-white/30 h-6 w-6 flex items-center justify-center text-xs font-bold flex-shrink-0">
            {abbreviation}
          </span>
        </div>
      </div>

      {/* Body with parameter summary */}
      <div className="px-3 py-2 bg-white">
        <div className="text-xs text-gray-600 line-clamp-2">{paramSummary}</div>
      </div>

      {/* Input Handles */}
      {isMergeLayer ? (
        // Merge layers (Concatenate) have multiple named input handles
        <>
          <Handle
            type="target"
            position={Position.Top}
            id="input-0"
            style={{ left: "30%", background: "#555" }}
            className="h-3 w-3 rounded-full border-2 border-white"
          />
          <Handle
            type="target"
            position={Position.Top}
            id="input-1"
            style={{ left: "70%", background: "#555" }}
            className="h-3 w-3 rounded-full border-2 border-white"
          />
        </>
      ) : (
        // Standard layers have a single centered input handle
        <Handle
          type="target"
          position={Position.Top}
          style={{ background: "#555" }}
          className="h-3 w-3 rounded-full border-2 border-white"
        />
      )}

      {/* Output Handle (all layers have one output) */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#555" }}
        className="h-3 w-3 rounded-full border-2 border-white"
      />
    </div>
  );
});

GenericLayerNode.displayName = "GenericLayerNode";

export default GenericLayerNode;
