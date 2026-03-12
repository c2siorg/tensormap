import { useState, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
<<<<<<< HEAD
  BackgroundVariant,
} from "reactflow";

import "reactflow/dist/style.css";

import Sidebar from "./Sidebar";
import NodePropertiesPanel from "./NodePropertiesPanel";
import ContextMenu from "./ContextMenu";

import InputNode from "./CustomNodes/InputNode/InputNode";
import DenseNode from "./CustomNodes/DenseNode/DenseNode";
import FlattenNode from "./CustomNodes/FlattenNode/FlattenNode";
import ConvNode from "./CustomNodes/ConvNode/ConvNode";
import BatchNormalizationNode from "./CustomNodes/BatchNormalizationNode/BatchNormalizationNode";

const nodeTypes = {
  input: InputNode,
  dense: DenseNode,
  flatten: FlattenNode,
  conv2d: ConvNode,
  dropout: DropoutNode,
  batchnorm: BatchNormalizationNode,
};

const defaultViewport = {
  x: 10,
  y: 15,
  zoom: 0.5,
};

function CanvasInner() {
=======
} from "reactflow";
import "reactflow/dist/style.css";

import { useRecoilState } from "recoil";

import Sidebar from "./Sidebar";
import NodePropertiesPanel from "./NodePropertiesPanel";
import ModelSummaryPanel from "./ModelSummaryPanel";
import { allModels } from "../../shared/atoms";

function Canvas() {
  const { projectId } = useParams();
  const [, setModelList] = useRecoilState(allModels);

  return (
    <CanvasInner
      projectId={projectId}
      setModelList={setModelList}
    />
  );
}

function CanvasInner({ projectId, setModelList }) {
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)
  const reactFlowWrapper = useRef(null);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
<<<<<<< HEAD

  const [contextMenu, setContextMenu] = useState({
    nodeId: null,
    x: 0,
    y: 0,
  });

=======
  const [modelSummary, setModelSummary] = useState(null);

  const [contextMenu, setContextMenu] = useState({
    nodeId: null,
    x: 0,
    y: 0,
  });

  const draftKey = `tensormap_draft_${projectId || "default"}`;

  const selectedNode =
    selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;

>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)
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

<<<<<<< HEAD
      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");
=======
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
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)

      if (!type || !reactFlowInstance) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      const newNode = {
        id: crypto.randomUUID(),
        type: type,
        position: position,
        data: { label: type + " node" },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const onNodeClick = useCallback((_, node) => {
    setSelectedNodeId(node.id);
  }, []);

  const onNodeContextMenu = useCallback((event, node) => {
    event.preventDefault();
    setContextMenu({
      nodeId: node.id,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu({ nodeId: null, x: 0, y: 0 });
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    closeContextMenu();
  }, [closeContextMenu]);

<<<<<<< HEAD
  const selectedNode = selectedNodeId
    ? nodes.find((n) => n.id === selectedNodeId)
    : null;

  return (
    <div className="flex gap-4">
      <Sidebar />

      <div className="flex-1 h-[62vh]" ref={reactFlowWrapper}>
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
          onNodeContextMenu={onNodeContextMenu}
          onPaneClick={onPaneClick}
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

      {contextMenu.nodeId && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={closeContextMenu}
        />
      )}

      <div className="w-64 shrink-0">
        <NodePropertiesPanel selectedNode={selectedNode} />
      </div>
    </div>
  );
}

function Canvas() {
  const { projectId } = useParams();

  return (
    <ReactFlowProvider>
      <CanvasInner key={projectId} />
    </ReactFlowProvider>
=======
  const onNodeContextMenu = useCallback((event, node) => {
    event.preventDefault();
    setContextMenu({
      nodeId: node.id,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

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
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)
  );
}

export default Canvas;
