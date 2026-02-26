import { useState, useRef, useCallback, useEffect } from "react";
import { useParams } from "react-router-dom";
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
import { Undo2, Redo2 } from "lucide-react";
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
import { getAllModels, getModelGraph, saveModel } from "../../services/ModelServices";
import { models as allModels } from "../../shared/atoms";
import { Button } from "@/components/ui/button";
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
  const [feedbackDialog, setFeedbackDialog] = useState({
    open: false,
    success: false,
    message: "",
    detail: "",
  });
  const defaultViewport = { x: 10, y: 15, zoom: 0.5 };

  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  const { takeSnapshot, undo, redo, canUndo, canRedo } = useUndoRedo(setNodes, setEdges, nodesRef, edgesRef);
  const [undoRedoState, setUndoRedoState] = useState({ canUndo: false, canRedo: false });

  const updateUndoRedoState = useCallback(() => {
    setUndoRedoState({ canUndo: canUndo(), canRedo: canRedo() });
  }, [canUndo, canRedo]);

  const isUndoRedoInProgress = useRef(false);

  const takeSnapshotAndUpdate = useCallback(
    (currentNodes, currentEdges) => {
      if (isUndoRedoInProgress.current) return;
      takeSnapshot(currentNodes, currentEdges);
      updateUndoRedoState();
    },
    [takeSnapshot, updateUndoRedoState],
  );

  const undoRef = useRef(undo);
  const redoRef = useRef(redo);
  const updateUndoRedoStateRef = useRef(updateUndoRedoState);

  useEffect(() => {
    undoRef.current = undo;
    redoRef.current = redo;
    updateUndoRedoStateRef.current = updateUndoRedoState;
  }, [undo, redo, updateUndoRedoState]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (isUndoRedoInProgress.current) return;

      const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
      const modKey = isMac ? event.metaKey : event.ctrlKey;

      if (modKey && event.key === "z" && !event.shiftKey) {
        event.preventDefault();
        event.stopPropagation();
        isUndoRedoInProgress.current = true;
        undoRef.current();
        updateUndoRedoStateRef.current();
        setTimeout(() => {
          isUndoRedoInProgress.current = false;
        }, 100);
      } else if (
        (modKey && event.key === "y") ||
        (modKey && event.shiftKey && event.key === "z") ||
        (modKey && event.shiftKey && event.key === "Z")
      ) {
        event.preventDefault();
        event.stopPropagation();
        isUndoRedoInProgress.current = true;
        redoRef.current();
        updateUndoRedoStateRef.current();
        setTimeout(() => {
          isUndoRedoInProgress.current = false;
        }, 100);
      }
    };

    document.addEventListener("keydown", handleKeyDown, true);
    return () => document.removeEventListener("keydown", handleKeyDown, true);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        const modelNames = await getAllModels(projectId);
        if (cancelled || !modelNames || modelNames.length === 0) return;

        const result = await getModelGraph(modelNames[0], projectId);
        if (cancelled || !result.success) return;

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

        setNodes(loadedNodes);
        setEdges(loadedEdges);
        setModelName(model_name);
      } catch (err) {
        logger.error("Failed to auto-load model:", err);
      }
    }
    loadModel();
    return () => {
      cancelled = true;
    };
  }, [projectId, setNodes, setEdges]);

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

  const onNodeDragStop = useCallback(() => {}, []);

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

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

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
          setModelList((prevList) => [
            ...prevList,
            {
              label: modelName + strings.MODEL_EXTENSION,
              value: modelName,
              key: prevList.length + 1,
            },
          ]);
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

  const handleUndo = useCallback(() => {
    if (isUndoRedoInProgress.current) return;
    isUndoRedoInProgress.current = true;
    undo();
    updateUndoRedoState();
    setTimeout(() => {
      isUndoRedoInProgress.current = false;
    }, 100);
  }, [undo, updateUndoRedoState]);

  const handleRedo = useCallback(() => {
    if (isUndoRedoInProgress.current) return;
    isUndoRedoInProgress.current = true;
    redo();
    updateUndoRedoState();
    setTimeout(() => {
      isUndoRedoInProgress.current = false;
    }, 100);
  }, [redo, updateUndoRedoState]);

  return (
    <>
      <FeedbackDialog
        open={feedbackDialog.open}
        onClose={closeFeedback}
        success={feedbackDialog.success}
        message={feedbackDialog.message}
        detail={feedbackDialog.detail}
      />
      <div className="flex gap-4">
        <ReactFlowProvider>
          <Sidebar />
          <div className="h-[62vh] flex-1" ref={reactFlowWrapper}>
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
              onNodeDragStart={onNodeDragStart}
              onNodeDragStop={onNodeDragStop}
              nodeTypes={nodeTypes}
              defaultViewport={defaultViewport}
            >
              <Panel position="top-left" className="flex gap-1">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleUndo}
                  disabled={!undoRedoState.canUndo}
                  title="Undo (Ctrl+Z)"
                  className="h-8 w-8"
                >
                  <Undo2 className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleRedo}
                  disabled={!undoRedoState.canRedo}
                  title="Redo (Ctrl+Y)"
                  className="h-8 w-8"
                >
                  <Redo2 className="h-4 w-4" />
                </Button>
              </Panel>
              <Controls />
              <Background
                id="1"
                gap={10}
                color="#e5e5e5"
                style={{ backgroundColor: "#fafafa" }}
                variant={BackgroundVariant.Dots}
              />
            </ReactFlow>
          </div>
          <div className="w-64 shrink-0">
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
    </>
  );
}

export default Canvas;
