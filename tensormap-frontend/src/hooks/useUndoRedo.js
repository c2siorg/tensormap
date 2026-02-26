import { useCallback, useRef } from "react";

const MAX_HISTORY_SIZE = 50;

/**
 * Custom hook for undo/redo functionality on ReactFlow canvas state.
 *
 * Uses a past/future two-stack pattern. Before each user action, the current
 * state is pushed to the past stack. Undo pops from past (pushing the current
 * state to future). Redo pops from future (pushing the current state to past).
 *
 * Rapid snapshots within 50 ms are deduplicated so that a single delete action
 * (which fires both node-remove and edge-remove) records only one entry.
 *
 * @param {Function} setNodes - ReactFlow setNodes function
 * @param {Function} setEdges - ReactFlow setEdges function
 * @param {React.MutableRefObject<Array>} nodesRef - Ref to current nodes
 * @param {React.MutableRefObject<Array>} edgesRef - Ref to current edges
 * @returns {{
 *   takeSnapshot: (nodes: Array, edges: Array) => void,
 *   undo: () => boolean,
 *   redo: () => boolean,
 *   canUndo: () => boolean,
 *   canRedo: () => boolean,
 * }}
 */
export default function useUndoRedo(setNodes, setEdges, nodesRef, edgesRef) {
  const pastRef = useRef([]);
  const futureRef = useRef([]);
  const lastSnapshotTimeRef = useRef(0);

  /**
   * Saves the given state to the past stack and clears the future stack.
   * Calls within 50 ms of each other are deduplicated to batch related
   * change events (e.g. node deletion + connected edge deletion).
   */
  const takeSnapshot = useCallback((nodes, edges) => {
    const now = Date.now();
    if (now - lastSnapshotTimeRef.current < 50) return;
    lastSnapshotTimeRef.current = now;

    futureRef.current = [];
    pastRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    if (pastRef.current.length > MAX_HISTORY_SIZE) {
      pastRef.current.shift();
    }
  }, []);

  /**
   * Undo: pushes the current canvas state to the future stack and restores
   * the most recent past state.
   * @returns {boolean} true if undo was performed
   */
  const undo = useCallback(() => {
    if (pastRef.current.length === 0) return false;

    futureRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodesRef.current)),
      edges: JSON.parse(JSON.stringify(edgesRef.current)),
    });

    const previous = pastRef.current.pop();
    setNodes(JSON.parse(JSON.stringify(previous.nodes)));
    setEdges(JSON.parse(JSON.stringify(previous.edges)));
    return true;
  }, [setNodes, setEdges, nodesRef, edgesRef]);

  /**
   * Redo: pushes the current canvas state to the past stack and restores
   * the most recent future state.
   * @returns {boolean} true if redo was performed
   */
  const redo = useCallback(() => {
    if (futureRef.current.length === 0) return false;

    pastRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodesRef.current)),
      edges: JSON.parse(JSON.stringify(edgesRef.current)),
    });

    const next = futureRef.current.pop();
    setNodes(JSON.parse(JSON.stringify(next.nodes)));
    setEdges(JSON.parse(JSON.stringify(next.edges)));
    return true;
  }, [setNodes, setEdges, nodesRef, edgesRef]);

  const canUndo = useCallback(() => pastRef.current.length > 0, []);
  const canRedo = useCallback(() => futureRef.current.length > 0, []);

  return { takeSnapshot, undo, redo, canUndo, canRedo };
}
