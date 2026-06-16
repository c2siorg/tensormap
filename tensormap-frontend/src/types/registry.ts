/**
 * Type definitions for the Layer Registry API.
 * These types must match the backend registry schema from app/layers/registry.py
 */

export type ParamType = "int" | "float" | "bool" | "enum";

export interface ParamSpec {
  name: string;
  type: ParamType;
  required: boolean;
  default: string | number | boolean | null;
  min?: number | null;
  max?: number | null;
  values?: string[] | null;
  description?: string;
}

export interface LayerSpec {
  type_key: string;
  display_name: string;
  category: string;
  keras_class: string;
  merge: boolean;
  description: string;
  params: Record<string, ParamSpec>;
}

export interface LayerRegistryResponse {
  categories: string[];
  layers: Record<string, LayerSpec[]>;
}

/**
 * Category color mappings for visual organization.
 * Separate maps to ensure Tailwind purge keeps all classes.
 */
export const CATEGORY_BG_COLORS: Record<string, string> = {
  core: "bg-blue-500",
  convolutional: "bg-purple-500",
  recurrent: "bg-green-500",
  regularization: "bg-orange-500",
  pooling: "bg-cyan-500",
  encoding: "bg-pink-500",
  utility: "bg-gray-500",
};

export const CATEGORY_BORDER_COLORS: Record<string, string> = {
  core: "border-blue-600",
  convolutional: "border-purple-600",
  recurrent: "border-green-600",
  regularization: "border-orange-600",
  pooling: "border-cyan-600",
  encoding: "border-pink-600",
  utility: "border-gray-600",
};

/**
 * Legacy combined color map - kept for backward compatibility but deprecated.
 * Use CATEGORY_BG_COLORS and CATEGORY_BORDER_COLORS instead.
 * @deprecated
 */
export const CATEGORY_COLORS: Record<string, string> = {
  core: "bg-blue-500 border-blue-600",
  convolutional: "bg-purple-500 border-purple-600",
  recurrent: "bg-green-500 border-green-600",
  regularization: "bg-orange-500 border-orange-600",
  pooling: "bg-cyan-500 border-cyan-600",
  encoding: "bg-pink-500 border-pink-600",
  utility: "bg-gray-500 border-gray-600",
};

/**
 * Two-letter abbreviations for layer types.
 * Used in node badges for quick visual identification.
 */
export const TYPE_ABBREVIATIONS: Record<string, string> = {
  input: "IN",
  dense: "DN",
  flatten: "FL",
  conv2d: "C2",
  maxpool2d: "MP",
  avgpool2d: "AP",
  globalavgpool2d: "GA",
  lstm: "LS",
  gru: "GR",
  simplernn: "RN",
  embedding: "EM",
  dropout: "DO",
  batchnorm: "BN",
  reshape: "RS",
  concatenate: "CT",
};

/**
 * Map old custom node types to new registry types for backward compatibility.
 * Used when loading models that were created before the registry-driven system.
 */
export const LEGACY_TYPE_MAP: Record<string, string> = {
  custominput: "input",
  customdense: "dense",
  customflatten: "flatten",
  customconv: "conv2d",
  customdropout: "dropout",
  custommaxpool: "maxpool2d",
  customglobalavgpool: "globalavgpool2d",
};
