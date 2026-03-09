import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import useUndoRedo from "./useUndoRedo";

/**
 * Helper: creates mock setters and stable refs, then renders the hook.
 * Returns { result, setNodes, setEdges, nodesRef, edgesRef }.
 */
function setup(initialNodes = [], initialEdges = []) {
  const setNodes = vi.fn((val) => {
    nodesRef.current = typeof val === "function" ? val(nodesRef.current) : val;
  });
  const setEdges = vi.fn((val) => {
    edgesRef.current = typeof val === "function" ? val(edgesRef.current) : val;
  });
  const nodesRef = { current: initialNodes };
  const edgesRef = { current: initialEdges };

  const { result } = renderHook(() =>
    useUndoRedo(setNodes, setEdges, nodesRef, edgesRef),
  );

  return { result, setNodes, setEdges, nodesRef, edgesRef };
}

describe("useUndoRedo", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ---------- empty-stack guards ----------

  it("undo on empty stack returns false and does not throw", () => {
    const { result } = setup();
    let ret;
    act(() => {
      ret = result.current.undo();
    });
    expect(ret).toBe(false);
  });

  it("redo on empty stack returns false and does not throw", () => {
    const { result } = setup();
    let ret;
    act(() => {
      ret = result.current.redo();
    });
    expect(ret).toBe(false);
  });

  // ---------- basic undo ----------

  it("undo restores the previous state", () => {
    const { result, setNodes, setEdges, nodesRef, edgesRef } = setup(
      [{ id: "b" }],
      [{ id: "e1" }],
    );

    // Snapshot the old state ([{id:'a'}], [])
    act(() => {
      result.current.takeSnapshot([{ id: "a" }], []);
    });

    // Simulate that the canvas moved to ({id:'b'}, {id:'e1'})
    // nodesRef/edgesRef already point there from setup

    act(() => {
      result.current.undo();
    });

    expect(setNodes).toHaveBeenCalledWith([{ id: "a" }]);
    expect(setEdges).toHaveBeenCalledWith([]);
  });

  // ---------- basic redo ----------

  it("redo restores the undone state", () => {
    const { result, setNodes, nodesRef } = setup([{ id: "a" }], []);

    // Take snapshot, advance state
    act(() => {
      result.current.takeSnapshot([{ id: "a" }], []);
    });
    nodesRef.current = [{ id: "b" }];

    // Undo → go back to a
    act(() => {
      result.current.undo();
    });
    // After undo, nodesRef simulates the restored state
    nodesRef.current = [{ id: "a" }];

    // Redo → go forward to b
    act(() => {
      result.current.redo();
    });
    expect(setNodes).toHaveBeenLastCalledWith([{ id: "b" }]);
  });

  // ---------- redo cleared after new action ----------

  it("clears redo history when a new action is taken after undo", () => {
    const { result, nodesRef, edgesRef } = setup([], []);

    act(() => {
      result.current.takeSnapshot([], []);
    });
    nodesRef.current = [{ id: "1" }];

    // Advance past dedup window
    vi.advanceTimersByTime(60);

    act(() => {
      result.current.takeSnapshot([{ id: "1" }], []);
    });
    nodesRef.current = [{ id: "2" }];
    edgesRef.current = [];

    // Undo
    act(() => {
      result.current.undo();
    });

    expect(result.current.canRedo).toBe(true);

    // Advance past dedup window, then new action
    vi.advanceTimersByTime(60);

    act(() => {
      result.current.takeSnapshot([{ id: "1" }], []);
    });

    // Redo should now be impossible
    expect(result.current.canRedo).toBe(false);
  });

  // ---------- MAX_HISTORY_SIZE ----------

  it("caps history at 50 entries (51st push drops the oldest)", () => {
    const { result, nodesRef, edgesRef } = setup([], []);

    for (let i = 0; i < 55; i++) {
      nodesRef.current = [{ id: String(i) }];
      edgesRef.current = [];

      act(() => {
        result.current.takeSnapshot([{ id: String(i) }], []);
      });
      // Advance past dedup window
      vi.advanceTimersByTime(60);
    }

    // canUndo should still be true
    expect(result.current.canUndo).toBe(true);

    // Undo 50 times — all should succeed
    let undoCount = 0;
    for (let i = 0; i < 55; i++) {
      let succeeded;
      act(() => {
        succeeded = result.current.undo();
      });
      if (succeeded) undoCount++;
    }
    expect(undoCount).toBe(50);
  });

  // ---------- dedup within 50 ms ----------

  it("deduplicates snapshots within 50 ms (only one entry added)", () => {
    const { result } = setup([], []);

    // Take two snapshots in rapid succession (< 50 ms)
    act(() => {
      result.current.takeSnapshot([{ id: "a" }], []);
      result.current.takeSnapshot([{ id: "b" }], []);
    });

    expect(result.current.canUndo).toBe(true);

    // Only one undo should be available
    let count = 0;
    act(() => {
      if (result.current.undo()) count++;
    });
    act(() => {
      if (result.current.undo()) count++;
    });
    expect(count).toBe(1);
  });

  // ---------- canUndo / canRedo reactive values ----------

  it("canUndo and canRedo correctly reflect stack state", async () => {
    const { result, nodesRef, edgesRef } = setup([{ id: "x" }], []);

    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);

    act(() => {
      result.current.takeSnapshot([{ id: "x" }], []);
    });

    expect(result.current.canUndo).toBe(true);
    expect(result.current.canRedo).toBe(false);

    act(() => {
      result.current.undo();
    });

    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(true);

    act(() => {
      result.current.redo();
    });

    expect(result.current.canUndo).toBe(true);
    expect(result.current.canRedo).toBe(false);
  });

  // ---------- future cleared even during dedup window ----------

  it("clears future stack even when snapshot is deduplicated", () => {
    const { result, nodesRef, edgesRef } = setup([], []);

    // Take snapshot and immediately undo
    act(() => {
      result.current.takeSnapshot([{ id: "a" }], []);
    });
    nodesRef.current = [{ id: "b" }];

    act(() => {
      result.current.undo();
    });

    expect(result.current.canRedo).toBe(true);

    // Take a new snapshot within the dedup window — future must still clear
    act(() => {
      result.current.takeSnapshot([{ id: "c" }], []);
    });

    expect(result.current.canRedo).toBe(false);
  });

  // ---------- safeClone fallback ----------

  it("handles non-cloneable data gracefully via JSON fallback", () => {
    const { result } = setup([], []);

    // Functions are not structuredClone-able but JSON.stringify drops them
    const nodesWithFn = [{ id: "a", data: { callback: () => {} } }];

    // Should not throw — falls back to JSON round-trip
    act(() => {
      result.current.takeSnapshot(nodesWithFn, []);
    });

    expect(result.current.canUndo).toBe(true);
  });

  // ---------- redo enforces MAX_HISTORY_SIZE on past stack ----------

  it("enforces MAX_HISTORY_SIZE on past stack during redo", () => {
    const { result, nodesRef, edgesRef } = setup([], []);

    // Fill past stack to 50
    for (let i = 0; i < 50; i++) {
      act(() => {
        result.current.takeSnapshot([{ id: String(i) }], []);
      });
      vi.advanceTimersByTime(60);
    }

    // Now undo once to create a redo entry
    nodesRef.current = [{ id: "current" }];
    edgesRef.current = [];
    act(() => {
      result.current.undo();
    });

    // Redo pushes to past — should still be capped at 50, not 51
    act(() => {
      result.current.redo();
    });

    // Verify by counting: undo should succeed at most 50 times
    let undoCount = 0;
    for (let i = 0; i < 55; i++) {
      let ok;
      act(() => {
        ok = result.current.undo();
      });
      if (ok) undoCount++;
    }
    expect(undoCount).toBeLessThanOrEqual(50);
  });

  // ---------- undo/redo abort when clone fails ----------

  it("undo aborts and returns false when current state cannot be cloned", () => {
    const { result, nodesRef } = setup([], []);

    // Take a valid snapshot
    act(() => {
      result.current.takeSnapshot([{ id: "a" }], []);
    });
    vi.advanceTimersByTime(60);

    // Stub both structuredClone and JSON.stringify to force safeClone → null
    const origClone = globalThis.structuredClone;
    const origStringify = JSON.stringify;
    globalThis.structuredClone = () => { throw new Error("clone fail"); };
    JSON.stringify = () => { throw new Error("stringify fail"); };

    let ret;
    act(() => {
      ret = result.current.undo();
    });

    // Restore
    globalThis.structuredClone = origClone;
    JSON.stringify = origStringify;

    // Undo should abort — state and stacks remain intact
    expect(ret).toBe(false);
    expect(result.current.canUndo).toBe(true);
    expect(result.current.canRedo).toBe(false);
  });

  it("redo aborts and returns false when current state cannot be cloned", () => {
    const { result, nodesRef } = setup([], []);

    // Take snapshot, advance, undo to create redo entry
    act(() => {
      result.current.takeSnapshot([{ id: "a" }], []);
    });
    vi.advanceTimersByTime(60);
    nodesRef.current = [{ id: "b" }];

    act(() => {
      result.current.undo();
    });
    expect(result.current.canRedo).toBe(true);

    // Stub both structuredClone and JSON.stringify to force safeClone → null
    const origClone = globalThis.structuredClone;
    const origStringify = JSON.stringify;
    globalThis.structuredClone = () => { throw new Error("clone fail"); };
    JSON.stringify = () => { throw new Error("stringify fail"); };

    let ret;
    act(() => {
      ret = result.current.redo();
    });

    // Restore
    globalThis.structuredClone = origClone;
    JSON.stringify = origStringify;

    // Redo should abort — redo stack remains intact
    expect(ret).toBe(false);
    expect(result.current.canRedo).toBe(true);
  });
});
