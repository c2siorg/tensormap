/**
 * Sidebar component with categorized, searchable layer palette.
 *
 * Phase 1 Week 2 - Fully registry-driven implementation.
 * Displays all 15 layer types organized by category with search functionality.
 */

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useLayerRegistry } from "../../hooks/useLayerRegistry";
import { CATEGORY_COLORS, TYPE_ABBREVIATIONS } from "../../types/registry";
import type { LayerSpec } from "../../types/registry";
/**
 * Category color mappings for visual organization.
 * Separate maps for background, border, and left-border to ensure Tailwind purge keeps them.
 */
const CATEGORY_BG_COLORS: Record<string, string> = {
  core: "bg-blue-500",
  convolutional: "bg-purple-500",
  recurrent: "bg-green-500",
  regularization: "bg-orange-500",
  pooling: "bg-cyan-500",
  encoding: "bg-pink-500",
  utility: "bg-gray-500",
};

const CATEGORY_BORDER_COLORS: Record<string, string> = {
  core: "border-blue-600",
  convolutional: "border-purple-600",
  recurrent: "border-green-600",
  regularization: "border-orange-600",
  pooling: "border-cyan-600",
  encoding: "border-pink-600",
  utility: "border-gray-600",
};

const CATEGORY_LEFT_BORDER_COLORS: Record<string, string> = {
  core: "border-l-blue-600",
  convolutional: "border-l-purple-600",
  recurrent: "border-l-green-600",
  regularization: "border-l-orange-600",
  pooling: "border-l-cyan-600",
  encoding: "border-l-pink-600",
  utility: "border-l-gray-600",
};

/**
 * Get the left border color class for a category.
 */
const getCategoryBorderClass = (category: string): string => {
  return CATEGORY_LEFT_BORDER_COLORS[category] || "border-l-gray-500";
};

/**
 * Get key params hint for display (e.g., "units, activation").
 */
const getParamsHint = (spec: LayerSpec): string => {
  const paramNames = Object.keys(spec.params).slice(0, 2);
  if (paramNames.length === 0) return "";
  return paramNames.join(", ");
};

/**
 * Skeleton loader for sidebar while registry loads.
 */
const SidebarSkeleton = () => (
  <Card className="h-full w-60 shrink-0">
    <CardHeader className="pb-3">
      <CardTitle className="text-sm">Layers</CardTitle>
    </CardHeader>
    <CardContent className="space-y-2">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="h-10 bg-gray-200 rounded animate-pulse" />
      ))}
    </CardContent>
  </Card>
);

/**
 * Individual draggable layer item.
 */
interface LayerItemProps {
  spec: LayerSpec;
  onDragStart: (event: React.DragEvent, typeKey: string) => void;
}

const LayerItem = ({ spec, onDragStart }: LayerItemProps) => {
  const borderClass = getCategoryBorderClass(spec.category);
  const abbreviation =
    TYPE_ABBREVIATIONS[spec.type_key] || spec.type_key.substring(0, 2).toUpperCase();
  const paramsHint = getParamsHint(spec);

  return (
    <div
      className={`cursor-grab active:cursor-grabbing rounded-md border border-l-4 ${borderClass} bg-white px-3 py-2 text-xs hover:shadow-md transition-shadow`}
      onDragStart={(e) => onDragStart(e, spec.type_key)}
      draggable
      title={spec.description}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="font-semibold truncate">{spec.display_name}</div>
          {paramsHint && <div className="text-gray-500 truncate text-[10px]">{paramsHint}</div>}
        </div>
        <div className="text-[10px] font-bold text-gray-600 flex-shrink-0">{abbreviation}</div>
      </div>
    </div>
  );
};

/**
 * Category section with collapse/expand functionality.
 */
interface CategorySectionProps {
  category: string;
  layers: LayerSpec[];
  isExpanded: boolean;
  onToggle: () => void;
  onDragStart: (event: React.DragEvent, typeKey: string) => void;
}

const CategorySection = ({
  category,
  layers,
  isExpanded,
  onToggle,
  onDragStart,
}: CategorySectionProps) => {
  return (
    <div className="space-y-1">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-2 py-1 hover:bg-gray-100 rounded text-xs font-semibold text-gray-700 uppercase transition-colors"
      >
        <span className="flex items-center gap-1">
          {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          {category}
        </span>
        <span className="text-gray-500">{layers.length}</span>
      </button>
      {isExpanded && (
        <div className="space-y-2 pl-1">
          {layers.map((spec) => (
            <LayerItem key={spec.type_key} spec={spec} onDragStart={onDragStart} />
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Main Sidebar component.
 */
function Sidebar() {
  const { registry, isLoading } = useLayerRegistry();
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set([
      "core",
      "convolutional",
      "pooling",
      "recurrent",
      "regularization",
      "encoding",
      "utility",
    ]),
  );

  const onDragStart = (event: React.DragEvent, typeKey: string) => {
    event.dataTransfer.setData("application/reactflow", typeKey);
    event.dataTransfer.effectAllowed = "move";
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  // Filter layers by search query
  const filteredLayers = useMemo(() => {
    if (!registry) return null;

    const query = searchQuery.toLowerCase().trim();
    if (!query) return registry.layers;

    // Filter each category's layers
    const filtered: Record<string, LayerSpec[]> = {};
    for (const [category, layers] of Object.entries(registry.layers)) {
      const matchingLayers = layers.filter(
        (spec) =>
          spec.display_name.toLowerCase().includes(query) ||
          spec.type_key.toLowerCase().includes(query) ||
          spec.description.toLowerCase().includes(query),
      );
      if (matchingLayers.length > 0) {
        filtered[category] = matchingLayers;
      }
    }
    return filtered;
  }, [registry, searchQuery]);

  // Calculate total layer count
  const totalCount = useMemo(() => {
    if (!filteredLayers) return 0;
    return Object.values(filteredLayers).reduce((sum, layers) => sum + layers.length, 0);
  }, [filteredLayers]);

  // Auto-expand all categories when searching
  const shouldExpandAll = searchQuery.trim() !== "";

  if (isLoading || !registry) {
    return <SidebarSkeleton />;
  }

  return (
    <Card className="h-full w-60 shrink-0 flex flex-col">
      <CardHeader className="pb-3 flex-shrink-0">
        <CardTitle className="text-sm flex items-center justify-between">
          <span>Layers</span>
          <span className="text-xs font-normal text-gray-500">({totalCount})</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden flex flex-col space-y-3">
        {/* Search Input */}
        <div className="relative flex-shrink-0">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-gray-400" />
          <Input
            type="text"
            placeholder="Search layers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-7 h-8 text-xs"
          />
        </div>

        {/* Layer Categories */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-2">
          {registry.categories.map((category) => {
            const layers = filteredLayers?.[category] || [];
            if (layers.length === 0) return null;

            return (
              <CategorySection
                key={category}
                category={category}
                layers={layers}
                isExpanded={shouldExpandAll || expandedCategories.has(category)}
                onToggle={() => toggleCategory(category)}
                onDragStart={onDragStart}
              />
            );
          })}

          {totalCount === 0 && (
            <div className="text-xs text-gray-500 text-center py-4">
              No layers match &quot;{searchQuery}&quot;
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default Sidebar;
