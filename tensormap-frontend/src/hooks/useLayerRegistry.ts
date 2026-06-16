/**
 * Custom hook for fetching and caching the layer registry from the backend.
 *
 * The registry is fetched once on mount and cached in module-level state,
 * preventing redundant API calls when multiple components use the registry.
 *
 * Usage:
 *   const { registry, isLoading, error } = useLayerRegistry();
 *   const layerSpec = getLayerSpec('dense');
 */

import { useState, useEffect } from "react";
import type { LayerRegistryResponse, LayerSpec } from "../types/registry";
import logger from "../shared/logger";

// Module-level cache - survives component unmounts
let cachedRegistry: LayerRegistryResponse | null = null;
let isFetching = false;
const listeners: Array<(registry: LayerRegistryResponse | null) => void> = [];

/**
 * Fetch the layer registry from the backend API.
 * Uses module-level caching to prevent duplicate requests.
 */
async function fetchRegistry(): Promise<LayerRegistryResponse> {
  if (cachedRegistry) {
    return cachedRegistry;
  }

  if (isFetching) {
    // Wait for ongoing fetch to complete
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        const idx = listeners.indexOf(handler);
        if (idx >= 0) listeners.splice(idx, 1);
      };
      const handler = (reg: LayerRegistryResponse | null) => {
        cleanup();
        if (reg) resolve(reg);
        else reject(new Error("Failed to fetch registry"));
      };
      listeners.push(handler);
    });
  }

  isFetching = true;

  try {
    const apiUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000/api/v1";
    const response = await fetch(`${apiUrl}/layers`);

    if (!response.ok) {
      throw new Error(`Failed to fetch layer registry: ${response.statusText}`);
    }

    const rawData: LayerSpec[] = await response.json();
    
    // Transform flat array into categorized structure
    const layersByCategory: Record<string, LayerSpec[]> = {};
    const categoriesSet = new Set<string>();
    
    for (const layer of rawData) {
      if (!layersByCategory[layer.category]) {
        layersByCategory[layer.category] = [];
      }
      layersByCategory[layer.category].push(layer);
      categoriesSet.add(layer.category);
    }
    
    const data: LayerRegistryResponse = {
      categories: Array.from(categoriesSet).sort(),
      layers: layersByCategory,
    };
    
    cachedRegistry = data;

    // Notify all waiting listeners
    listeners.forEach((listener) => listener(data));
    listeners.length = 0;

    return data;
  } catch (error) {
    logger.error("Error fetching layer registry:", error);
    // Notify all waiting listeners of failure
    listeners.forEach((listener) => listener(null));
    listeners.length = 0;
    throw error;
  } finally {
    isFetching = false;
  }
}

/**
 * React hook for accessing the layer registry.
 *
 * @returns Object containing:
 *   - registry: The complete registry data (null if not loaded)
 *   - isLoading: True while fetching
 *   - error: Error object if fetch failed
 */
export function useLayerRegistry() {
  const [registry, setRegistry] = useState<LayerRegistryResponse | null>(cachedRegistry);
  const [isLoading, setIsLoading] = useState<boolean>(!cachedRegistry);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (cachedRegistry) {
      setRegistry(cachedRegistry);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    fetchRegistry()
      .then((data) => {
        setRegistry(data);
        setError(null);
      })
      .catch((err) => {
        setError(err);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, []);

  return { registry, isLoading, error };
}

/**
 * Get a specific layer specification by its type key.
 *
 * @param typeKey - The unique identifier for the layer (e.g., 'dense', 'lstm')
 * @returns LayerSpec if found, null otherwise
 */
export function getLayerSpec(typeKey: string): LayerSpec | null {
  if (!cachedRegistry) {
    return null;
  }

  // Search through all categories
  for (const layerSpecs of Object.values(cachedRegistry.layers)) {
    const spec = layerSpecs.find((s) => s.type_key === typeKey);
    if (spec) {
      return spec;
    }
  }

  return null;
}

/**
 * Get all layer specifications as a flat array.
 * Useful for searching and filtering across all layers.
 */
export function getAllLayerSpecs(): LayerSpec[] {
  if (!cachedRegistry) {
    return [];
  }

  return Object.values(cachedRegistry.layers).flat();
}

/**
 * Clear the module-level cache.
 * Primarily used for testing.
 */
export function clearRegistryCache(): void {
  cachedRegistry = null;
}
