import { useState, useRef, useCallback, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Trash2, Undo2, Redo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  Panel,
} from "reactflow";
import { useRecoilState } from "recoil";
import * as strings from "../../constants/Strings";
import logger from "../../shared/logger";
import FeedbackDialog from "../shared/FeedbackDialog";
import "reactflow/dist/style.css";
import InputNode from "./CustomNodes/InputNode/InputNode";
import DenseNode from "./CustomNodes/DenseNode/DenseNode";
import FlattenNode from "./CustomNodes/FlattenNode/FlattenNode";
import ConvNode from "./CustomNodes/ConvNode/ConvNode";
import Sidebar from "./Sidebar";
import NodePropertiesPanel from "./NodePropertiesPanel";
import { canSaveModel, generateModelJSON } from "./Helpers";
import ModelSummaryPanel from "./ModelSummaryPanel";
import { getAllModels, getModelGraph, saveModel } from "../../services/ModelServices";
import { models as allModels } from "../../shared/atoms";
import ContextMenu from "./ContextMenu";
import useUndoRedo from "../../hooks/useUndoRedo";

const nodeTypes = {
  custominput: InputNode,
  customdense: DenseNode,
  customflatten: FlattenNode,
  customconv: ConvNode,
};

function Canvas() {
  const { projectId } = useParams();
  const reactFlowWrapper = useRef(null);
  const [, setModelList] = useRecoilState(allModels);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [modelName, setModelName] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [modelSummary, setModelSummary] = useState(null);
  const [feedbackDialog, setFeedbackDialog] = useState({
    open: false,
    success: false,
    message: "",
    detail: "",
  });
  const [contextMenu, setContextMenu] = useState({ nodeId: null, x: 0, y: 0 });
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const defaultViewport = { x: 10, y: 15, zoom: 0.5 };

  const draftKey = `tensormap_draft_${projectId || "default"}`;
  const isLoaded = useRef(false);
  const [hasDraft, setHasDraft] = useState(() => {
    try {
      return !!localStorage.getItem(draftKey);
    } catch (e) {
      return false;
    }
  });

  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  const { takeSnapshot, undo, redo, canUndo, canRedo } = useUndoRedo(
    setNodes,
    setEdges,
    nodesRef,
    edgesRef,
  );
  const [undoRedoState, setUndoRedoState] = useState({ canUndo: false, canRedo: false });

  const updateUndoRedoState = useCallback(() => {
    setUndoRedoState({ canUndo: canUndo(), canRedo: canRedo() });
  }, [canUndo, canRedo]);

  const undoRedoCounter = useRef(0);

  const takeSnapshotAndUpdate = useCallback(
    (currentNodes, currentEdges) => {
      if (undoRedoCounter.current > 0) return;
      takeSnapshot(currentNodes, currentEdges);
      updateUndoRedoState();
    },
    [takeSnapshot, updateUndoRedoState],
  );

  const performUndo = useCallback(() => {
    undoRedoCounter.current += 1;
    undo();
    updateUndoRedoState();
    undoRedoCounter.current -= 1;
  }, [undo, updateUndoRedoState]);

  const performRedo = useCallback(() => {
    undoRedoCounter.current += 1;
    redo();
    updateUndoRedoState();
    undoRedoCounter.current -= 1;
  }, [redo, updateUndoRedoState]);

  const undoRef = useRef(performUndo);
  const redoRef = useRef(performRedo);

  useEffect(() => {
    undoRef.current = performUndo;
    redoRef.current = performRedo;
  }, [performUndo, performRedo]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      const el = document.activeElement;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || el?.isContentEditable) return;

      const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
      const modKey = isMac ? event.metaKey : event.ctrlKey;

      if (modKey && event.key === "z" && !event.shiftKey) {
        event.preventDefault();
        event.stopPropagation();
        undoRef.current();
      } else if (
        (modKey && event.key === "y") ||
        (modKey && event.shiftKey && event.key === "z") ||
        (modKey && event.shiftKey && event.key === "Z")
      ) {
        event.preventDefault();
        event.stopPropagation();
        redoRef.current();
      }
    };

    document.addEventListener("keydown", handleKeyDown, true);
    return () => document.removeEventListener("keydown", handleKeyDown, true);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      // 1. Try loading from draft first
      try {
        const draftStr = localStorage.getItem(draftKey);
        if (draftStr) {
          const draft = JSON.parse(draftStr);
          if (draft.nodes?.length > 0 || draft.edges?.length > 0 || draft.modelName) {
            if (!cancelled) {
              setNodes(draft.nodes || []);
              setEdges(draft.edges || []);
              setModelName(draft.modelName || "");
              setHasDraft(true);
              isLoaded.current = true;
            }
            return;
          }
        }
      } catch (e) {
        logger.error("Failed to load draft:", e);
      }

      // 2. Fallback to loading from DB
      try {
        const modelObjects = await getAllModels(projectId);
        if (cancelled || !modelObjects || modelObjects.length === 0) {
          if (!cancelled) isLoaded.current = true;
          return;
        }

        const result = await getModelGraph(modelObjects[0].model_name, projectId);
        if (cancelled || !result.success) {
          if (!cancelled) isLoaded.current = true;
          return;
        }

        const { graph, model_name } = result.data;

        const loadedNodes = (graph.nodes || []).map((node, i) => ({
          id: node.id,
          type: node.type,
          position: node.position || { x: 100, y: i * 200 },
          data: { label: `${node.type} node`, params: node.data?.params || {} },
        }));

        const loadedEdges = (graph.edges || []).map((edge) => ({
          id: edge.id || `e-${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
        }));

        if (!cancelled) {
          setNodes(loadedNodes);
          setEdges(loadedEdges);
          setModelName(model_name);
          isLoaded.current = true;

          setModelList(
            modelObjects.map((m, i) => ({
              label: m.model_name + strings.MODEL_EXTENSION,
              value: m.model_name,
              id: m.id,
              key: i,
            })),
          );
        }
      } catch (err) {
        logger.error("Failed to auto-load model:", err);
        if (!cancelled) isLoaded.current = true;
      }
    }
    loadModel();
    return () => {
      cancelled = true;
    };
  }, [projectId, setNodes, setEdges, setModelList, draftKey]);

  // Handle debounced saving of draft
  useEffect(() => {
    if (!isLoaded.current) return;

    const timer = setTimeout(() => {
      if (nodes.length === 0 && edges.length === 0 && !modelName) {
        try {
          localStorage.removeItem(draftKey);
        } catch (e) {
          logger.error("Failed to remove draft:", e);
        }
        setHasDraft(false);
        return;
      }

      try {
        localStorage.setItem(draftKey, JSON.stringify({ nodes, edges, modelName }));
        setHasDraft(true);
      } catch (e) {
        logger.error("Failed to save draft:", e);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [nodes, edges, modelName, draftKey]);

  const handleDiscardDraft = useCallback(() => {
    try {
      localStorage.removeItem(draftKey);
    } catch (e) {
      logger.error("Failed to remove draft:", e);
    }
    setHasDraft(false);
    setNodes([]);
    setEdges([]);
    setModelName("");
  }, [draftKey, setNodes, setEdges]);

  const onConnect = useCallback(
    (params) => {
      takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges, takeSnapshotAndUpdate],
  );

  const handleNodesChange = useCallback(
    (changes) => {
      const hasRemoval = changes.some((change) => change.type === "remove");
      if (hasRemoval) {
        takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
      }
      onNodesChange(changes);
    },
    [onNodesChange, takeSnapshotAndUpdate],
  );

  const handleEdgesChange = useCallback(
    (changes) => {
      const hasRemoval = changes.some((change) => change.type === "remove");
      if (hasRemoval) {
        takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
      }
      onEdgesChange(changes);
    },
    [onEdgesChange, takeSnapshotAndUpdate],
  );

  const onNodeDragStart = useCallback(() => {
    takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
  }, [takeSnapshotAndUpdate]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const modelData =
    reactFlowInstance === null ? { nodes: [], edges: [] } : reactFlowInstance.toObject();

  const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;

  const onNodeClick = useCallback((_event, node) => {
    setSelectedNodeId(node.id);
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu({ nodeId: null, x: 0, y: 0 });
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    closeContextMenu();
  }, [closeContextMenu]);

  const onNodeContextMenu = useCallback((event, node) => {
    event.preventDefault();
    setContextMenu({ nodeId: node.id, x: event.clientX, y: event.clientY });
  }, []);

  const duplicateNode = useCallback(() => {
    takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
    setNodes((nds) => {
      const source = nds.find((n) => n.id === contextMenu.nodeId);
      if (!source) return nds;
      const duplicate = {
        id: crypto.randomUUID(),
        type: source.type,
        position: { x: source.position.x + 50, y: source.position.y + 50 },
        data: { label: source.data.label, params: { ...source.data.params } },
      };
      return nds.concat(duplicate);
    });
    closeContextMenu();
  }, [contextMenu.nodeId, setNodes, closeContextMenu, takeSnapshotAndUpdate]);

  const onNodeUpdate = useCallback(
    (nodeId, newParams) => {
      takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return { ...node, data: { ...node.data, params: newParams } };
          }
          return node;
        }),
      );
    },
    [setNodes, takeSnapshotAndUpdate],
  );

  const closeFeedback = () => {
    setFeedbackDialog((prev) => ({ ...prev, open: false }));
  };

  const handleClearAll = useCallback(() => {
    takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
    setNodes([]);
    setEdges([]);
    setModelName("");
    setSelectedNodeId(null);
    setClearConfirmOpen(false);
  }, [setNodes, setEdges, takeSnapshotAndUpdate]);

  const modelSaveHandler = () => {
    const data = {
      model: {
        ...generateModelJSON(reactFlowInstance.toObject()),
        model_name: modelName,
      },
      model_name: modelName,
      project_id: projectId || null,
    };

    saveModel(data)
      .then((resp) => {
        if (resp.success) {
          setModelSummary(resp.data?.summary || null);
          try {
            localStorage.removeItem(draftKey);
            setHasDraft(false);
          } catch (e) {
            logger.error("Failed to clear draft on save:", e);
          }
          getAllModels(projectId)
            .then((modelObjects) => {
              setModelList(
                modelObjects.map((m, i) => ({
                  label: m.model_name + strings.MODEL_EXTENSION,
                  value: m.model_name,
                  id: m.id,
                  key: i,
                })),
              );
            })
            .catch(() => {});
        }
        setFeedbackDialog({
          open: true,
          success: resp.success,
          message: resp.success
            ? strings.MODEL_VALIDATION_MODAL_MESSAGE
            : strings.PROCESS_FAIL_MODEL_MESSAGE,
          detail: resp.message,
        });
      })
      .catch((error) => {
        logger.error(error);
        setModelSummary(null);
        setFeedbackDialog({
          open: true,
          success: false,
          message: strings.PROCESS_FAIL_MODEL_MESSAGE,
          detail: error.message || "Model save failed",
        });
      });
  };

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");

      if (typeof type === "undefined" || !type) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const defaultParams = {
        custominput: { "dim-1": "", "dim-2": "", "dim-3": "" },
        customdense: { units: "", activation: "" },
        customflatten: {},
        customconv: {
          filter: "",
          padding: "valid",
          activation: "none",
          strideX: "",
          strideY: "",
          kernelX: "",
          kernelY: "",
        },
      };

      const newNode = {
        id: crypto.randomUUID(),
        type,
        position,
        data: { label: `${type} node`, params: defaultParams[type] || {} },
      };

      takeSnapshotAndUpdate(nodesRef.current, edgesRef.current);
      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes, takeSnapshotAndUpdate],
  );

  return (
    <>
      <FeedbackDialog
        open={feedbackDialog.open}
        onClose={closeFeedback}
        success={feedbackDialog.success}
        message={feedbackDialog.message}
        detail={feedbackDialog.detail}
      />
      <Dialog open={clearConfirmOpen} onOpenChange={setClearConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear canvas</DialogTitle>
            <DialogDescription>
              This will remove all {nodes.length} node{nodes.length !== 1 ? "s" : ""} and their
              connections. You can undo this with Ctrl+Z.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setClearConfirmOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleClearAll}>
              Clear All
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <div className="flex gap-4">
        <ReactFlowProvider>
          <Sidebar />
          <div className="flex flex-col flex-1 gap-2">
            <div className="flex justify-end">
              <Button
                variant="destructive"
                size="sm"
                disabled={nodes.length === 0}
                onClick={() => setClearConfirmOpen(true)}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Clear All
              </Button>
            </div>
            <div className="min-w-0 h-[62vh] flex-1 rounded-md border" ref={reactFlowWrapper}>
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={handleNodesChange}
                onEdgesChange={handleEdgesChange}
                onConnect={onConnect}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onNodeClick={onNodeClick}
                onPaneClick={onPaneClick}
                onNodeContextMenu={onNodeContextMenu}
                onNodeDragStart={onNodeDragStart}
                nodeTypes={nodeTypes}
                defaultViewport={defaultViewport}
              >
                <Panel position="top-left" className="flex gap-1">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={performUndo}
                    disabled={!undoRedoState.canUndo}
                    title="Undo (Ctrl+Z)"
                    className="h-8 w-8"
                  >
                    <Undo2 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={performRedo}
                    disabled={!undoRedoState.canRedo}
                    title="Redo (Ctrl+Y)"
                    className="h-8 w-8"
                  >
                    <Redo2 className="h-4 w-4" />
                  </Button>
                </Panel>
                <Controls />
                {hasDraft && (
                  <Panel position="top-right">
                    <Button variant="destructive" onClick={handleDiscardDraft}>
                      Discard Draft
                    </Button>
                  </Panel>
                )}
                <Background
                  id="1"
                  gap={10}
                  color="#e5e5e5"
                  style={{ backgroundColor: "#fafafa" }}
                  variant={BackgroundVariant.Dots}
                />
              </ReactFlow>
            </div>
          </div>
          {contextMenu.nodeId && (
            <ContextMenu
              x={contextMenu.x}
              y={contextMenu.y}
              onDuplicate={duplicateNode}
              onClose={closeContextMenu}
            />
          )}
          <div className="w-72 shrink-0">
            <NodePropertiesPanel
              selectedNode={selectedNode || null}
              modelName={modelName}
              onModelNameChange={setModelName}
              onSave={modelSaveHandler}
              canSave={canSaveModel(modelName, modelData)}
              onNodeUpdate={onNodeUpdate}
            />
          </div>
        </ReactFlowProvider>
      </div>
      <ModelSummaryPanel summary={modelSummary} onClose={() => setModelSummary(null)} />
    </>
  );
}

export default Canvas;
