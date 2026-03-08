import { useState, useRef, useCallback } from "react";
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
  const reactFlowWrapper = useRef(null);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);

  const [contextMenu, setContextMenu] = useState({
    nodeId: null,
    x: 0,
    y: 0,
  });

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

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");

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
  );
}

export default Canvas;
