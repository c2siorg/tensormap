import { useState, useRef, useCallback, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Trash2 } from "lucide-react";
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
import Toast from "./Toast";
import { canSaveModel, generateModelJSON } from "./Helpers";
import ModelSummaryPanel from "./ModelSummaryPanel";
import { getAllModels, getModelGraph, saveModel } from "../../services/ModelServices";
import { models as allModels } from "../../shared/atoms";
import ContextMenu from "./ContextMenu";

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
  const [toastMessage, setToastMessage] = useState(null);
  const connectionErrorRef = useRef(null);
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

  // Auto-load the project's first saved model or draft on mount
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

          // Populate the global model list from the fetched models
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

  const onConnect = useCallback((params) => {
    // Mark connection as successful so mouseup doesn't show a false error
    connectionDragRef.current = null;
    setEdges((eds) => addEdge(params, eds));
  }, [setEdges]);

  const isValidConnection = useCallback(
    (connection) => {
      // Must connect output to input
      const sourceIsOut = connection.sourceHandle && connection.sourceHandle.endsWith("_out");
      const targetIsIn = connection.targetHandle && connection.targetHandle.endsWith("_in");

      if (!sourceIsOut || !targetIsIn) {
        connectionErrorRef.current = "Cannot connect handles of the same type";
        return false;
      }

      // Prevent self-loops
      if (connection.source === connection.target) {
        connectionErrorRef.current = "Cannot connect a node to itself";
        return false;
      }

      // Prevent multiple incoming edges
      const hasIncoming = edges.some((e) => e.target === connection.target);
      if (hasIncoming) {
        connectionErrorRef.current = "Target node already has an incoming connection";
        return false;
      }

      // Prevent duplicate connections
      const isDuplicate = edges.some(
        (e) => e.source === connection.source && e.target === connection.target
      );
      if (isDuplicate) {
        connectionErrorRef.current = "Connection already exists";
        return false;
      }

      // Prevent cycles (e.g. A→B already exists, reject B→A)
      // BFS from target: if we can reach source through existing edges, adding
      // source→target would form a cycle.
      const visited = new Set();
      const queue = [connection.target];
      while (queue.length > 0) {
        const curr = queue.shift();
        if (curr === connection.source) {
          connectionErrorRef.current = "This connection would create a cycle";
          return false;
        }
        if (visited.has(curr)) continue;
        visited.add(curr);
        edges.filter((e) => e.source === curr).forEach((e) => queue.push(e.target));
      }

      connectionErrorRef.current = null;
      return true;
    },
    [edges]
  );

  // Store the source node/handle when dragging starts so we can validate on mouseup
  const connectionDragRef = useRef(null);

  const onConnectStart = useCallback((_event, { nodeId, handleId, handleType }) => {
    connectionErrorRef.current = null;
    connectionDragRef.current = { nodeId, handleId, handleType };
  }, []);

  // onConnectEnd is required to prevent ReactFlow warnings
  const onConnectEnd = useCallback(() => { }, []);

  // On mouseup, determine what element the user dropped onto and validate.
  // This approach works even when ReactFlow silently swallows the connection
  // attempt (e.g. dropping on an already-connected handle).
  useEffect(() => {
    const handleMouseUp = (event) => {
      const drag = connectionDragRef.current;
      if (!drag) return; // Not in a connection drag, or valid connection already consumed
      connectionDragRef.current = null;

      // isValidConnection already set an error (e.g. self-loop, wrong handle type)
      if (connectionErrorRef.current) {
        setToastMessage(connectionErrorRef.current);
        connectionErrorRef.current = null;
        return;
      }

      // Walk up from the drop target to find the ReactFlow handle.
      // ReactFlow v11 puts data-handleid and data-nodeid directly on handle divs.
      let el = event.target;
      let targetHandleId = null;
      let targetNodeId = null;
      while (el && el !== document.body) {
        const hid = el.getAttribute && el.getAttribute('data-handleid');
        if (hid !== null && hid !== undefined) {
          targetHandleId = hid;
          targetNodeId = el.getAttribute('data-nodeid');
          break;
        }
        el = el.parentElement;
      }

      if (!targetHandleId || !targetNodeId) return;
      if (drag.handleType !== 'source') return;

      // Self-loop (belt-and-suspenders — also caught by isValidConnection)
      if (targetNodeId === drag.nodeId) {
        setToastMessage('Cannot connect a node to itself');
        return;
      }

      // Only valid drop direction: _out → _in
      if (!drag.handleId?.endsWith('_out') || !targetHandleId.endsWith('_in')) return;

      // Duplicate edge
      const isDuplicate = edges.some(
        (e) => e.source === drag.nodeId && e.target === targetNodeId
      );
      if (isDuplicate) {
        setToastMessage('Connection already exists between these nodes');
        return;
      }

      // Target already has an incoming edge
      const hasIncoming = edges.some((e) => e.target === targetNodeId);
      if (hasIncoming) {
        setToastMessage('Target node already has an incoming connection');
      }
    };

    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [edges]);

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
  }, [contextMenu.nodeId, setNodes, closeContextMenu]);

  const onNodeUpdate = useCallback(
    (nodeId, newParams) => {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return { ...node, data: { ...node.data, params: newParams } };
          }
          return node;
        }),
      );
    },
    [setNodes],
  );

  const closeFeedback = () => {
    setFeedbackDialog((prev) => ({ ...prev, open: false }));
  };

  const handleClearAll = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setModelName("");
    setSelectedNodeId(null);
    setClearConfirmOpen(false);
  }, [setNodes, setEdges]);

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
          // Re-fetch the model list so the new entry has its DB id
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
            .catch(() => { });
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

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes],
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
      <Toast message={toastMessage} onClose={() => setToastMessage(null)} />
      <Dialog open={clearConfirmOpen} onOpenChange={setClearConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear canvas</DialogTitle>
            <DialogDescription>
              This will remove all {nodes.length} node{nodes.length !== 1 ? "s" : ""} and their
              connections. This action cannot be undone.
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
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                isValidConnection={isValidConnection}
                onConnectStart={onConnectStart}
                onConnectEnd={onConnectEnd}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onNodeClick={onNodeClick}
                onPaneClick={onPaneClick}
                onNodeContextMenu={onNodeContextMenu}
                nodeTypes={nodeTypes}
                defaultViewport={defaultViewport}
              >
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
