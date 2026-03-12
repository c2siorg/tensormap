import { useState, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
} from "reactflow";
import "reactflow/dist/style.css";
import { useRecoilState } from "recoil";
import Sidebar from "./Sidebar";
import NodePropertiesPanel from "./NodePropertiesPanel";
import ModelSummaryPanel from "./ModelSummaryPanel";
import { allModels } from "../../shared/atoms";

const defaultViewport = { x: 10, y: 15, zoom: 0.5 };

function Canvas() {
  const { projectId } = useParams();
  const [, setModelList] = useRecoilState(allModels);
  const [contextMenu, setContextMenu] = useState({
    nodeId: null,
    x: 0,
    y: 0,
  });

  return (
    <CanvasInner
      projectId={projectId}
      setModelList={setModelList}
      contextMenu={contextMenu}
      setContextMenu={setContextMenu}
    />
  );
}

function CanvasInner({ setContextMenu }) {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [modelName, setModelName] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [modelSummary, setModelSummary] = useState(null);
  const [, setReactFlowInstance] = useState(null);

  const selectedNode =
    selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData("application/reactflow");
      if (!type) return;
      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      };
      const newNode = {
        id: crypto.randomUUID(),
        type,
        position,
        data: {
          label: `${type} node`,
          params: {},
        },
      };
      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes]
  );

  const onNodeClick = useCallback((_event, node) => {
    setSelectedNodeId(node.id);
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu({ nodeId: null, x: 0, y: 0 });
  }, [setContextMenu]);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    closeContextMenu();
  }, [closeContextMenu]);

  const onNodeContextMenu = useCallback(
    (event, node) => {
      event.preventDefault();
      setContextMenu({
        nodeId: node.id,
        x: event.clientX,
        y: event.clientY,
      });
    },
    [setContextMenu]
  );

  const onNodeUpdate = useCallback(
    (nodeId, newParams) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, params: newParams } }
            : node
        )
      );
    },
    [setNodes]
  );

  return (
    <>
      <ReactFlowProvider>
        <div className="flex gap-4">
          <Sidebar />
          <div className="flex flex-col flex-1 gap-2">
            <div
              className="min-w-0 h-[62vh] flex-1 rounded-md border"
              ref={reactFlowWrapper}
            >
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onPaneClick={onPaneClick}
                onNodeClick={onNodeClick}
                onNodeContextMenu={onNodeContextMenu}
                defaultViewport={defaultViewport}
                fitView
              >
                <Controls />
                <Background />
              </ReactFlow>
            </div>
          </div>
          <div className="w-72 shrink-0">
            <NodePropertiesPanel
              selectedNode={selectedNode || null}
              modelName={modelName}
              onModelNameChange={setModelName}
              onNodeUpdate={onNodeUpdate}
            />
          </div>
        </div>
      </ReactFlowProvider>
      <ModelSummaryPanel
        summary={modelSummary}
        onClose={() => setModelSummary(null)}
      />
    </>
  );
}

export default Canvas;
