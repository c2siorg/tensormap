import { useCallback, useRef, useState } from "react";

const MAX_HISTORY_SIZE = 50;

/**
 * Safely deep-clone a value using structuredClone, falling back to
 * JSON round-trip when the value contains non-cloneable data (functions,
 * DOM nodes, Symbols, etc.).
 *
 * @param {*} value - The value to clone
 * @returns {*} A deep copy of value, or null if cloning fails entirely
 */
function safeClone(value) {
  try {
    return structuredClone(value);
  } catch {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (e) {
      console.warn("[useUndoRedo] Could not clone state; snapshot skipped.", e);
      return null;
    }
  }
}

/**
 * Push an entry to a stack, enforcing a maximum size.
 *
 * @param {Array} stack - The history stack (past or future)
 * @param {object} entry - The {nodes, edges} snapshot to push
 * @param {number} max - Maximum allowed stack length
 */
function cappedPush(stack, entry, max) {
  stack.push(entry);
  if (stack.length > max) stack.shift();
}

/**
 * Custom hook for undo/redo functionality on ReactFlow canvas state.
 *
 * Uses a past/future two-stack pattern. Before each user action, the current
 * state is pushed to the past stack. Undo pops from past (pushing the current
 * state to future). Redo pops from future (pushing the current state to past).
 *
 * Rapid snapshots within 50 ms are deduplicated so that a single delete action
 * (which fires both node-remove and edge-remove) records only one entry.
 * The future (redo) stack is always cleared when a new action is taken, even
 * if the snapshot itself is deduplicated.
 *
 * @param {Function} setNodes - ReactFlow setNodes function
 * @param {Function} setEdges - ReactFlow setEdges function
 * @param {React.MutableRefObject<Array>} nodesRef - Ref to current nodes
 * @param {React.MutableRefObject<Array>} edgesRef - Ref to current edges
 * @returns {{
 *   takeSnapshot: (nodes: Array, edges: Array) => void,
 *   undo: () => boolean,
 *   redo: () => boolean,
 *   canUndo: boolean,
 *   canRedo: boolean,
 * }}
 */
export default function useUndoRedo(setNodes, setEdges, nodesRef, edgesRef) {
  const pastRef = useRef([]);
  const futureRef = useRef([]);
  const lastSnapshotTimeRef = useRef(0);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);

  /** Synchronise the reactive canUndo/canRedo state with the ref stacks. */
  const syncState = useCallback(() => {
    setCanUndo(pastRef.current.length > 0);
    setCanRedo(futureRef.current.length > 0);
  }, []);

  /**
   * Saves the given state to the past stack and clears the future stack.
   * Calls within 50 ms of each other are deduplicated to batch related
   * change events (e.g. node deletion + connected edge deletion).
   *
   * The future stack is **always** cleared when a new action is taken,
   * even if the snapshot itself is deduplicated (prevents stale redo
   * after undo → new action within the dedup window).
   */
  const takeSnapshot = useCallback(
    (nodes, edges) => {
      // Always invalidate the redo stack on a new intentional action
      futureRef.current = [];

      const now = Date.now();
      if (now - lastSnapshotTimeRef.current < 50) {
        syncState();
        return;
      }
      lastSnapshotTimeRef.current = now;

      const clonedNodes = safeClone(nodes);
      const clonedEdges = safeClone(edges);
      if (clonedNodes === null || clonedEdges === null) {
        syncState();
        return;
      }

      cappedPush(pastRef.current, { nodes: clonedNodes, edges: clonedEdges }, MAX_HISTORY_SIZE);
      syncState();
    },
    [syncState],
  );

  /**
   * Undo: pushes the current canvas state to the future stack and restores
   * the most recent past state. Data is already deep-cloned when stored, so
   * no re-cloning is needed when restoring.
   * @returns {boolean} true if undo was performed
   */
  const undo = useCallback(() => {
    if (pastRef.current.length === 0) return false;

    const clonedNodes = safeClone(nodesRef.current);
    const clonedEdges = safeClone(edgesRef.current);
    if (clonedNodes === null || clonedEdges === null) {
      // Cannot preserve current state for redo — abort to avoid data loss
      console.warn("[useUndoRedo] undo aborted: failed to clone current state");
      return false;
    }

    cappedPush(futureRef.current, { nodes: clonedNodes, edges: clonedEdges }, MAX_HISTORY_SIZE);

    const previous = pastRef.current.pop();
    setNodes(previous.nodes);
    setEdges(previous.edges);
    syncState();
    return true;
  }, [setNodes, setEdges, nodesRef, edgesRef, syncState]);

  /**
   * Redo: pushes the current canvas state to the past stack and restores
   * the most recent future state. Data is already deep-cloned when stored, so
   * no re-cloning is needed when restoring.
   * @returns {boolean} true if redo was performed
   */
  const redo = useCallback(() => {
    if (futureRef.current.length === 0) return false;

    const clonedNodes = safeClone(nodesRef.current);
    const clonedEdges = safeClone(edgesRef.current);
    if (clonedNodes === null || clonedEdges === null) {
      // Cannot preserve current state for undo — abort to avoid data loss
      console.warn("[useUndoRedo] redo aborted: failed to clone current state");
      return false;
    }

    cappedPush(pastRef.current, { nodes: clonedNodes, edges: clonedEdges }, MAX_HISTORY_SIZE);

    const next = futureRef.current.pop();
    setNodes(next.nodes);
    setEdges(next.edges);
    syncState();
    return true;
  }, [setNodes, setEdges, nodesRef, edgesRef, syncState]);

  return { takeSnapshot, undo, redo, canUndo, canRedo };
}
