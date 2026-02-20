import { useState, useRef, useCallback, useEffect } from "react";
import { Grid, Form, Button } from "semantic-ui-react";
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  getNodesBounds,
  getViewportForBounds,
} from "reactflow";
import { toPng } from "html-to-image";
import { useRecoilState } from "recoil";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import * as strings from "../../constants/Strings";
import "reactflow/dist/style.css";
import InputNode from "./CustomNodes/InputNode/InputNode";
import DenseNode from "./CustomNodes/DenseNode/DenseNode";
import FlattenNode from "./CustomNodes/FlattenNode/FlattenNode";
import ConvNode from "./CustomNodes/ConvNode/ConvNode";
import KeyboardShortcutsHelp from "./KeyboardShortcutsHelp";

import Sidebar from "./Sidebar";
import PropertiesBar from "../PropertiesBar/PropertiesBar";
import "./Canvas.css";
import {
  enableValidateButton,
  generateModelJSON,
  InitialFormState,
} from "./Helpers";
import { validateModel } from "../../services/ModelServices";
import { models as allModels } from "../../shared/atoms";

let id = 0;
const getId = () => `dndnode_${id++}`;
const nodeTypes = {
  custominput: InputNode,
  customdense: DenseNode,
  customflatten: FlattenNode,
  customconv: ConvNode,
};

const cloneFlowState = (flowNodes, flowEdges) => ({
  nodes: JSON.parse(JSON.stringify(flowNodes)),
  edges: JSON.parse(JSON.stringify(flowEdges)),
});

function Canvas() {
  const reactFlowWrapper = useRef(null);
  const [, setModelList] = useRecoilState(allModels);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [history, setHistory] = useState({ past: [], future: [] });
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [formState, setFormState] = useState(InitialFormState);
  const defaultViewport = { x: 10, y: 15, zoom: 0.5 };

  const pushHistory = useCallback(() => {
    setHistory((prevState) => ({
      past: [...prevState.past, cloneFlowState(nodes, edges)],
      future: [],
    }));
  }, [nodes, edges]);

  const onNodesChange = useCallback(
    (changes) => {
      const hasStructuralChange = changes.some(
        (change) => change.type !== "select",
      );
      if (hasStructuralChange) {
        pushHistory();
      }
      setNodes((currentNodes) => applyNodeChanges(changes, currentNodes));
    },
    [pushHistory],
  );

  const onEdgesChange = useCallback(
    (changes) => {
      const hasStructuralChange = changes.some(
        (change) => change.type !== "select",
      );
      if (hasStructuralChange) {
        pushHistory();
      }
      setEdges((currentEdges) => applyEdgeChanges(changes, currentEdges));
    },
    [pushHistory],
  );

  const onConnect = useCallback(
    (params) => {
      pushHistory();
      setEdges((currentEdges) => addEdge(params, currentEdges));
    },
    [pushHistory],
  );

  const undoFlow = useCallback(() => {
    setHistory((prevState) => {
      if (prevState.past.length === 0) {
        return prevState;
      }

      const previous = prevState.past[prevState.past.length - 1];
      setNodes(previous.nodes);
      setEdges(previous.edges);

      return {
        past: prevState.past.slice(0, -1),
        future: [cloneFlowState(nodes, edges), ...prevState.future],
      };
    });
  }, [nodes, edges]);

  const redoFlow = useCallback(() => {
    setHistory((prevState) => {
      if (prevState.future.length === 0) {
        return prevState;
      }

      const next = prevState.future[0];
      setNodes(next.nodes);
      setEdges(next.edges);

      return {
        past: [...prevState.past, cloneFlowState(nodes, edges)],
        future: prevState.future.slice(1),
      };
    });
  }, [nodes, edges]);

  const deleteSelectedElements = useCallback(() => {
    const selectedNodeIds = nodes
      .filter((node) => node.selected)
      .map((node) => node.id);
    const hasSelectedEdges = edges.some((edge) => edge.selected);
    if (selectedNodeIds.length === 0 && !hasSelectedEdges) {
      return;
    }

    pushHistory();
    setNodes((currentNodes) => currentNodes.filter((node) => !node.selected));
    setEdges((currentEdges) =>
      currentEdges.filter(
        (edge) =>
          !edge.selected &&
          !selectedNodeIds.includes(edge.source) &&
          !selectedNodeIds.includes(edge.target),
      ),
    );
  }, [nodes, edges, pushHistory]);

  // Supports duplicating multiple selected nodes at once
  const duplicateSelectedNodes = useCallback(() => {
    const selectedNodes = nodes.filter((node) => node.selected);
    if (selectedNodes.length === 0) {
      return;
    }

    const newNodes = selectedNodes.map((selectedNode) => ({
      ...JSON.parse(JSON.stringify(selectedNode)),
      id: getId(),
      selected: false,
      position: {
        x: selectedNode.position.x + 40,
        y: selectedNode.position.y + 40,
      },
      data: {
        ...selectedNode.data,
        label: `${selectedNode.data.label} copy`,
      },
    }));

    pushHistory();
    setNodes((currentNodes) =>
      currentNodes
        .map((node) => ({ ...node, selected: false }))
        .concat(newNodes),
    );
  }, [nodes, pushHistory]);

  const clearCanvas = useCallback(() => {
    if (nodes.length === 0 && edges.length === 0) return;
    if (!window.confirm("Clear the entire canvas? This can be undone.")) return;
    pushHistory();
    setNodes([]);
    setEdges([]);
    toast.info("Canvas cleared. Press Ctrl+Z to undo.");
  }, [nodes, edges, pushHistory]);

  const exportToPng = useCallback(() => {
    if (!reactFlowWrapper.current) return;

    const nodesBounds = getNodesBounds(nodes);
    const viewport = getViewportForBounds(nodesBounds, 1280, 720, 0.5, 2, 0.1);

    const rfElement = reactFlowWrapper.current.querySelector(".react-flow__viewport");
    if (!rfElement) {
      toast.error("Could not find canvas to export.");
      return;
    }

    toPng(rfElement, {
      backgroundColor: "#3c3c3c",
      width: 1280,
      height: 720,
      style: {
        width: 1280,
        height: 720,
        transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
      },
    })
      .then((dataUrl) => {
        const a = document.createElement("a");
        a.setAttribute("download", "tensormap-model.png");
        a.setAttribute("href", dataUrl);
        a.click();
        toast.success("Canvas exported as PNG!");
      })
      .catch(() => {
        toast.error("Failed to export canvas.");
      });
  }, [nodes, reactFlowWrapper]);

  useEffect(() => {
    const keyDownHandler = (event) => {
      const targetTag = (event.target?.tagName || "").toLowerCase();
      const isEditableTarget =
        targetTag === "input" ||
        targetTag === "textarea" ||
        event.target?.isContentEditable;
      if (isEditableTarget) {
        return;
      }

      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && key === "z") {
        event.preventDefault();
        if (event.shiftKey) {
          redoFlow();
        } else {
          undoFlow();
        }
      } else if ((event.metaKey || event.ctrlKey) && key === "d") {
        event.preventDefault();
        duplicateSelectedNodes();
      } else if (key === "delete" || key === "backspace") {
        event.preventDefault();
        deleteSelectedElements();
      }
    };

    window.addEventListener("keydown", keyDownHandler);
    return () => window.removeEventListener("keydown", keyDownHandler);
  }, [deleteSelectedElements, duplicateSelectedNodes, redoFlow, undoFlow]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const modelData =
    reactFlowInstance === null ? {} : reactFlowInstance.toObject();

  const modelValidateHandler = () => {
    const data = {
      code: {
        dataset: {
          file_id: formState.selectedFile,
          target_field: formState.targetField,
          training_split: formState.trainTestRatio,
        },
        dl_model: {
          model_name: formState.modalName,
          optimizer: formState.optimizer,
          metric: formState.metric,
          epochs: formState.epochCount,
        },
        problem_type_id: formState.problemType,
      },
      model: {
        ...generateModelJSON(reactFlowInstance.toObject()),
        model_name: formState.modalName,
      },
    };

    validateModel(data)
      .then((validationResp) => {
        if (validationResp.success) {
          setModelList((prevList) => [
            ...prevList,
            {
              text: formState.modalName + strings.MODEL_EXTENSION,
              value: formState.modalName,
              key: prevList.length + 1,
            },
          ]);
          toast.success(strings.MODEL_VALIDATION_MODAL_MESSAGE || "Model validated successfully!");
        } else {
          toast.error(validationResp.message || strings.PROCESS_FAIL_MODEL_MESSAGE);
        }
        setFormState((prevState) => ({
          ...prevState,
          modelValidatedSuccessfully: validationResp.success,
          modalContent: validationResp.message,
        }));
      })
      .catch((error) => {
        console.error(error);
        // Spread to preserve all form fields on error (bug fix)
        setFormState((prevState) => ({
          ...prevState,
          modelValidatedSuccessfully: false,
          modalContent: error.message,
        }));
        toast.error(error.message || strings.PROCESS_FAIL_MODEL_MESSAGE);
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

      // screenToFlowPosition replaces deprecated project() in ReactFlow v11+
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });
      let newNode;
      if (type === "custominput") {
        newNode = {
          id: getId(),
          type,
          position,
          data: {
            label: `${type} node`,
            params: { "dim-1": "", "dim-2": "", "dim-3": "" },
          },
        };
      } else if (type === "customdense") {
        newNode = {
          id: getId(),
          type,
          position,
          data: {
            label: `${type} node`,
            params: { units: "", activation: "default" },
          },
        };
      } else if (type === "customflatten") {
        newNode = {
          id: getId(),
          type,
          position,
          data: { label: `${type} node`, params: { "dim-x": "", "dim-y": "" } },
        };
      } else if (type === "customconv") {
        newNode = {
          id: getId(),
          type,
          position,
          data: {
            label: `${type} node`,
            params: {
              filter: "",
              padding: "valid",
              activation: "none",
              strideX: "",
              strideY: "",
              kernelX: "",
              kernelY: "",
            },
          },
        };
      } else {
        newNode = {
          id: getId(),
          type,
          position,
          data: { label: `${type} node` },
        };
      }
      pushHistory();
      setNodes((nds) => nds.concat(newNode));
    },
    [pushHistory, reactFlowInstance],
  );

  return (
    <>
      <ToastContainer
        position="top-right"
        autoClose={4000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        pauseOnHover
        theme="dark"
      />
      <Grid celled="internally">
        <Grid.Row>
          <Grid.Column width={13}>
            {/* Canvas Stats Bar */}
            <div className="canvas-stats-bar">
              <span>Nodes: <strong>{nodes.length}</strong></span>
              <span>Edges: <strong>{edges.length}</strong></span>
              <span>History: <strong>{history.past.length}</strong> steps</span>
            </div>
            <div className="dndflow">
              <ReactFlowProvider>
                <Sidebar />
                <div className="reactflow-wrapper" ref={reactFlowWrapper}>
                  <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onInit={setReactFlowInstance}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    nodeTypes={nodeTypes}
                    defaultViewport={defaultViewport}
                  >
                    <Controls />
                    <MiniMap
                      nodeStrokeWidth={3}
                      zoomable
                      pannable
                      style={{ background: "#2b2b2b" }}
                      nodeColor="#4a90e2"
                    />
                    <Background
                      id="1"
                      gap={10}
                      color="#454545"
                      style={{ backgroundColor: "#3c3c3c" }}
                      variant={BackgroundVariant.Lines}
                    />
                  </ReactFlow>
                  <KeyboardShortcutsHelp />
                </div>
              </ReactFlowProvider>
            </div>
          </Grid.Column>
          <Grid.Column width={3}>
            <PropertiesBar formState={formState} setFormState={setFormState} />
            <Form>
              <Form.Field>
                <Button
                  color="blue"
                  size="small"
                  onClick={undoFlow}
                  disabled={history.past.length === 0}
                  style={{ marginTop: "8px", marginLeft: "15%" }}
                >
                  Undo
                </Button>
                <Button
                  color="teal"
                  size="small"
                  onClick={redoFlow}
                  disabled={history.future.length === 0}
                  style={{ marginTop: "8px", marginLeft: "8px" }}
                >
                  Redo
                </Button>
              </Form.Field>
              <Form.Field>
                <Button
                  color="orange"
                  size="small"
                  onClick={duplicateSelectedNodes}
                  style={{ marginTop: "6px", marginLeft: "15%" }}
                >
                  Duplicate Selected
                </Button>
              </Form.Field>
              <Form.Field>
                <Button
                  color="red"
                  size="small"
                  onClick={deleteSelectedElements}
                  style={{ marginTop: "6px", marginLeft: "15%" }}
                >
                  Delete Selected
                </Button>
              </Form.Field>
              <Form.Field>
                <Button
                  color="grey"
                  size="small"
                  onClick={clearCanvas}
                  disabled={nodes.length === 0 && edges.length === 0}
                  style={{ marginTop: "6px", marginLeft: "15%" }}
                >
                  Clear All
                </Button>
              </Form.Field>
              <Form.Field>
                <Button
                  color="violet"
                  size="small"
                  onClick={exportToPng}
                  disabled={nodes.length === 0}
                  style={{ marginTop: "6px", marginLeft: "15%" }}
                >
                  Export PNG
                </Button>
              </Form.Field>
              <Form.Field>
                <Button
                  color="green"
                  size="medium"
                  style={{ marginTop: "10px", marginLeft: "15%" }}
                  onClick={modelValidateHandler}
                  disabled={enableValidateButton(formState, modelData)}
                >
                  Validate Model
                </Button>
              </Form.Field>
            </Form>
          </Grid.Column>
        </Grid.Row>
      </Grid>
    </>
  );
}

export default Canvas;
