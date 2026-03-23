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
import GenericLayerNode from "./CustomNodes/GenericLayerNode";
import { getLayerRegistry } from "../../services/ModelServices";
import Sidebar from "./Sidebar";
import NodePropertiesPanel from "./NodePropertiesPanel";
import { canSaveModel } from "./Helpers";
import ModelSummaryPanel from "./ModelSummaryPanel";
import { getAllModels, getModelGraph, saveModel } from "../../services/ModelServices";
import { models as allModels } from "../../shared/atoms";
import ContextMenu from "./ContextMenu";

const nodeTypes = {
  genericLayer: GenericLayerNode,
  custominput: GenericLayerNode,
  customdense: GenericLayerNode,
  customconv: GenericLayerNode,
  customflatten: GenericLayerNode,
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
  const [layerRegistry, setLayerRegistry] = useState({});
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const defaultViewport = { x: 10, y: 15, zoom: 0.5 };

  useEffect(() => {
    getLayerRegistry().then(setLayerRegistry);
  }, []);

  const draftKey = `tensormap_draft_${projectId || "default"}`;
  const isLoaded = useRef(false);
  const [hasDraft, setHasDraft] = useState(() => {
    try {
      return !!localStorage.getItem(draftKey);
    } catch (e) {
      return false;
    }
  });

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
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

        // FIX 1: Ensure registry is preserved when loading from DB
        const loadedNodes = (graph.nodes || []).map((node, i) => ({
          id: node.id,
          type: node.type,
          position: node.position || { x: 100, y: i * 200 },
          data: {
            label: node.data?.label || `${node.type} node`,
            params: node.data?.params || {},
            registry: node.data?.registry || {},
          },
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

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const modelData =
    reactFlowInstance === null ? { nodes: [], edges: [] } : reactFlowInstance.toObject();
  const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;

  const onNodeClick = useCallback((_event, node) => setSelectedNodeId(node.id), []);
  const closeContextMenu = useCallback(() => setContextMenu({ nodeId: null, x: 0, y: 0 }), []);
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

      // FIX 2: Ensure registry is preserved when duplicating nodes
      const duplicate = {
        id: crypto.randomUUID(),
        type: source.type,
        position: { x: source.position.x + 50, y: source.position.y + 50 },
        data: {
          label: source.data.label,
          params: { ...source.data.params },
          registry: source.data.registry,
        },
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

  const closeFeedback = () => setFeedbackDialog((prev) => ({ ...prev, open: false }));

  const handleClearAll = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setModelName("");
    setSelectedNodeId(null);
    setClearConfirmOpen(false);
  }, [setNodes, setEdges]);

  const modelSaveHandler = () => {
    // Bypass the cached Helpers.js entirely and map the payload right here
    const rawData = reactFlowInstance.toObject();
    const cleanNodes = rawData.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        params: node.data.params,
        registry: node.data.registry,
      },
    }));

    const cleanEdges = rawData.edges.map((edge) => ({
      source: edge.source,
      target: edge.target,
    }));

    const data = {
      model: {
        nodes: cleanNodes,
        edges: cleanEdges,
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
      const layerKey = event.dataTransfer.getData("application/reactflow");
      if (typeof layerKey === "undefined" || !layerKey || !layerRegistry[layerKey]) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const layerConfig = layerRegistry[layerKey];
      const defaultParams = {};
      Object.entries(layerConfig.params || {}).forEach(([key, paramConfig]) => {
        defaultParams[key] = paramConfig.default !== undefined ? paramConfig.default : "";
      });

      const newNode = {
        id: crypto.randomUUID(),
        type: "genericLayer",
        position,
        data: {
          label: layerConfig.display_name,
          params: defaultParams,
          registry: layerConfig,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes, layerRegistry],
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
          {/* YOUR dynamic sidebar */}
          <Sidebar registry={layerRegistry} />

          {/* THEIR new layout wrappers and Clear All button */}
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
