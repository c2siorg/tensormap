/**
 * Canvas minimap colors for the bespoke node types.
 *
 * Values mirror the `node-*` Tailwind tokens in `tailwind.config.js` that the
 * custom node headers render with, so the minimap and the canvas cannot drift.
 * @module
 */

export const MINIMAP_NODE_COLORS = {
  custominput: "rgb(105, 172, 61)", // green
  customdense: "rgb(43, 161, 255)", // blue
  customflatten: "rgb(247, 173, 20)", // amber
  customconv: "rgb(255, 128, 43)", // orange
  customdropout: "rgb(220, 80, 80)", // red
  custommaxpool: "rgb(34, 182, 176)", // teal
  customglobalavgpool: "rgb(139, 92, 246)", // violet
};

/** Registry-driven `GenericLayerNode` types share this neutral slate. */
export const MINIMAP_FALLBACK_COLOR = "rgb(148, 163, 184)";

export function getMiniMapNodeColor(node) {
  return MINIMAP_NODE_COLORS[node?.type] ?? MINIMAP_FALLBACK_COLOR;
}
