/**
 * GenericLayerNode - Universal node component for all layer types.
 * 
 * TODO: Phase 1 Week 2 — wire registry data from GET /layers endpoint
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
 */

import { memo } from 'react';
import { Handle, Position } from 'reactflow';

/**
 * Layer specification from the backend registry.
 * This will be fetched from GET /layers in Phase 1 Week 2.
 */
interface LayerSpec {
  type_key: string;
  display_name: string;
  category: string;
  keras_class: string;
  params: Record<string, any>;
  merge: boolean;
  description: string;
}

interface GenericLayerNodeProps {
  data: {
    layerType: string;
    label: string;
    params: Record<string, any>;
    registrySpec?: LayerSpec;
  };
  selected: boolean;
}

/**
 * Get Tailwind background color class based on layer category.
 */
const getCategoryColor = (category: string): string => {
  const colorMap: Record<string, string> = {
    core: 'bg-blue-500',
    convolutional: 'bg-purple-500',
    recurrent: 'bg-green-500',
    regularization: 'bg-orange-500',
    pooling: 'bg-cyan-500',
    encoding: 'bg-pink-500',
    utility: 'bg-gray-500',
  };
  return colorMap[category] || 'bg-gray-500';
};

/**
 * Generate a two-letter abbreviation for the layer type.
 * Examples: Dense -> DN, LSTM -> LS, Conv2D -> CV
 */
const getLayerAbbreviation = (displayName: string): string => {
  // Remove numbers and special characters, take first two letters
  const cleaned = displayName.replace(/[0-9]/g, '').replace(/[^a-zA-Z]/g, '');
  return cleaned.substring(0, 2).toUpperCase();
};

/**
 * Format parameter summary for display.
 * Shows key parameters in a compact format (e.g., "units: 64, activation: relu")
 */
const formatParamSummary = (params: Record<string, any>): string => {
  const entries = Object.entries(params)
    .filter(([key]) => key !== 'layer_type') // Exclude discriminator field
    .slice(0, 2); // Show max 2 params to keep it compact

  if (entries.length === 0) return 'No parameters';

  return entries
    .map(([key, value]) => {
      // Format value based on type
      if (typeof value === 'boolean') return `${key}: ${value ? 'yes' : 'no'}`;
      if (typeof value === 'number') return `${key}: ${value}`;
      if (typeof value === 'string') return `${key}: ${value}`;
      return `${key}: ${JSON.stringify(value)}`;
    })
    .join(', ');
};

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
const GenericLayerNode = memo(({ data, selected }: GenericLayerNodeProps) => {
  const { layerType, label, params, registrySpec } = data;
  
  // Use registry spec if available, otherwise fall back to defaults
  const category = registrySpec?.category || 'core';
  const displayName = registrySpec?.display_name || label || layerType;
  const isMergeLayer = registrySpec?.merge || false;
  
  const categoryColor = getCategoryColor(category);
  const abbreviation = getLayerAbbreviation(displayName);
  const paramSummary = formatParamSummary(params);

  return (
    <div
      className={`
        rounded-lg border-2 bg-white shadow-md transition-all
        ${selected ? 'border-blue-600 shadow-lg' : 'border-gray-300'}
        min-w-[200px] max-w-[280px]
      `}
    >
      {/* Header with category color */}
      <div className={`${categoryColor} rounded-t-md px-3 py-2 text-white`}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold">{displayName}</span>
          <span className="rounded bg-white/20 px-2 py-0.5 text-xs font-bold">
            {abbreviation}
          </span>
        </div>
      </div>

      {/* Body with parameter summary */}
      <div className="px-3 py-2">
        <div className="text-xs text-gray-600">
          {paramSummary}
        </div>
      </div>

      {/* Input Handles */}
      {isMergeLayer ? (
        // Merge layers (Concatenate) have multiple named input handles
        <>
          <Handle
            type="target"
            position={Position.Top}
            id="input-0"
            style={{ left: '30%', background: '#555' }}
            className="h-3 w-3 rounded-full border-2 border-white"
          />
          <Handle
            type="target"
            position={Position.Top}
            id="input-1"
            style={{ left: '70%', background: '#555' }}
            className="h-3 w-3 rounded-full border-2 border-white"
          />
        </>
      ) : (
        // Standard layers have a single centered input handle
        <Handle
          type="target"
          position={Position.Top}
          style={{ background: '#555' }}
          className="h-3 w-3 rounded-full border-2 border-white"
        />
      )}

      {/* Output Handle (all layers have one output) */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#555' }}
        className="h-3 w-3 rounded-full border-2 border-white"
      />
    </div>
  );
});

GenericLayerNode.displayName = 'GenericLayerNode';

export default GenericLayerNode;
