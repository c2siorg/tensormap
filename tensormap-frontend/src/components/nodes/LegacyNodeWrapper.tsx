/**
 * LegacyNodeWrapper - Backward compatibility wrapper for existing node types.
 *
 * This component maps old node data shapes (custominput, customdense, etc.) to the new
 * registry-driven GenericLayerNode format. This allows the existing saved models
 * and canvas states to continue working while we migrate to the new system.
 *
 * Usage:
 *   const CustomDenseNode = createLegacyWrapper('dense');
 *
 * Old node types and their registry equivalents:
 *   custominput      -> input
 *   customdense      -> dense
 *   customflatten    -> flatten
 *   customconv       -> conv2d
 *   customdropout    -> dropout
 *   custommaxpool    -> maxpool2d
 *   customglobalavgpool -> globalavgpool2d
 */

import { memo } from "react";
import { NodeProps } from "reactflow";
import GenericLayerNode from "./GenericLayerNode";
import { LEGACY_TYPE_MAP } from "../../types/registry";

interface LegacyNodeData {
  label?: string;
  params: Record<string, any>;
}
/**
 * Factory function to create legacy wrapper components.
 *
 * @param oldType - The old custom node type (e.g., 'customdense')
 * @returns A component compatible with the old node type
 */
export function createLegacyWrapper(oldType: string) {
  const LegacyWrapper = memo((props: NodeProps<LegacyNodeData>) => {
    const registryType = LEGACY_TYPE_MAP[oldType] || oldType;

    // Map old data format to new format
    const newData = {
      layerType: registryType,
      params: props.data.params,
      label: props.data.label,
    };

    return <GenericLayerNode {...props} data={newData} />;
  });

  LegacyWrapper.displayName = `LegacyWrapper(${oldType})`;
  return LegacyWrapper;
}

// Pre-created wrappers for existing node types
export const LegacyInputNode = createLegacyWrapper("custominput");
export const LegacyDenseNode = createLegacyWrapper("customdense");
export const LegacyFlattenNode = createLegacyWrapper("customflatten");
export const LegacyConvNode = createLegacyWrapper("customconv");
export const LegacyDropoutNode = createLegacyWrapper("customdropout");
export const LegacyMaxPoolNode = createLegacyWrapper("custommaxpool");
export const LegacyGlobalAvgPoolNode = createLegacyWrapper("customglobalavgpool");
