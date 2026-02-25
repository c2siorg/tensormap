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
import { getAllModels, getModelGraph, saveModel } from "../../services/ModelServices";
import { models as allModels } from "../../shared/atoms";

const nodeTypes = {
  custominput: InputNode,
  customdense: DenseNode,
  customflatten: FlattenNode,
  customconv: ConvNode,
};

const nodeDescriptions = {
  custominput: "Defines the shape and format of the input data.",
  customdense: "Applies a learned linear transformation to the input.",
  customflatten: "Reduces spatial dimensions to a 1D vector.",
  customconv: "Applies a convolution filter to extract spatial features.",
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
  const [tooltip, setTooltip] = useState({ show: false, text: "", x: 0, y: 0 });
  const hoverTimeoutRef = useRef(null);
  const [feedbackDialog, setFeedbackDialog] = useState({
    open: false,
    success: false,
    message: "",
    detail: "",
  });
  const defaultViewport = { x: 10, y: 15, zoom: 0.5 };

  // Auto-load the project's first saved model on mount
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

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

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

  const onNodeMouseEnter = useCallback((event, node) => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    const description = nodeDescriptions[node.type];
    if (description) {
      hoverTimeoutRef.current = setTimeout(() => {
        setTooltip({
          show: true,
          text: description,
          x: event.clientX,
          y: event.clientY - 15, // Offset slightly above the mouse
        });
      }, 250);
    }
  }, []);

  const onNodeMouseLeave = useCallback(() => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    setTooltip({ show: false, text: "", x: 0, y: 0 });
  }, []);

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    };
  }, []);

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

      {tooltip.show && (
        <div
          className="fixed z-50 pointer-events-none px-3 py-2 text-sm text-white bg-gray-800 rounded shadow-lg max-w-xs transition-opacity duration-200"
          style={{ top: tooltip.y, left: tooltip.x, transform: "translate(-50%, -100%)" }}
        >
          {tooltip.text}
        </div>
      )}

      <div className="flex gap-4">
        <ReactFlowProvider>
          <Sidebar />
          <div className="h-[62vh] flex-1" ref={reactFlowWrapper}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onInit={setReactFlowInstance}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              onNodeMouseEnter={onNodeMouseEnter}
              onNodeMouseLeave={onNodeMouseLeave}
              nodeTypes={nodeTypes}
              defaultViewport={defaultViewport}
            >
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
