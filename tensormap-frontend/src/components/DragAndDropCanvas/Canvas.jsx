import { useState, useRef, useCallback } from "react";
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
import { Button } from "@/components/ui/button";
import * as strings from "../../constants/Strings";
import FeedbackDialog from "../shared/FeedbackDialog";
import "reactflow/dist/style.css";
import InputNode from "./CustomNodes/InputNode/InputNode";
import DenseNode from "./CustomNodes/DenseNode/DenseNode";
import FlattenNode from "./CustomNodes/FlattenNode/FlattenNode";
import ConvNode from "./CustomNodes/ConvNode/ConvNode";

import Sidebar from "./Sidebar";
import PropertiesBar from "../PropertiesBar/PropertiesBar";
import "./Canvas.css";
import { enableValidateButton, generateModelJSON, InitialFormState } from "./Helpers";
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

function Canvas() {
  const reactFlowWrapper = useRef(null);
  const [, setModelList] = useRecoilState(allModels);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [formState, setFormState] = useState(InitialFormState);
  const defaultViewport = { x: 10, y: 15, zoom: 0.5 };
  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const modelData = reactFlowInstance === null ? {} : reactFlowInstance.toObject();

  const closeModel = () => {
    setFormState((prevState) => ({ ...prevState, modalOpen: false }));
  };
  const openModel = () => {
    setFormState((prevState) => ({ ...prevState, modalOpen: true }));
  };

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
        }
        setFormState((prevState) => ({
          ...prevState,
          modelValidatedSuccessfully: validationResp.success,
          modalContent: validationResp.message,
        }));
        openModel();
      })
      .catch((error) => {
        console.error(error);
        setFormState({
          modelValidatedSuccessfully: false,
          modalContent: error.message,
        });
        openModel();
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
      let newNode;
      if (type === "custominput") {
        newNode = {
          id: getId(),
          type,
          position,
          data: { label: `${type} node`, params: { "dim-1": "", "dim-2": "", "dim-3": "" } },
        };
      } else if (type === "customdense") {
        newNode = {
          id: getId(),
          type,
          position,
          data: { label: `${type} node`, params: { units: "", activation: "default" } },
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
      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes],
  );

  return (
    <>
      <FeedbackDialog
        open={formState.modalOpen}
        onClose={closeModel}
        success={formState.modelValidatedSuccessfully}
        message={
          formState.modelValidatedSuccessfully
            ? strings.MODEL_VALIDATION_MODAL_MESSAGE
            : strings.PROCESS_FAIL_MODEL_MESSAGE
        }
        detail={formState.modalContent}
      />
      <div className="flex gap-4">
        <div className="flex-1">
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
                  <Background
                    id="1"
                    gap={10}
                    color="#454545"
                    style={{ backgroundColor: "#3c3c3c" }}
                    variant={BackgroundVariant.Lines}
                  />
                </ReactFlow>
              </div>
            </ReactFlowProvider>
          </div>
        </div>
        <div className="w-64 shrink-0">
          <PropertiesBar formState={formState} setFormState={setFormState} />
          <Button
            className="mt-4 w-full"
            onClick={modelValidateHandler}
            disabled={enableValidateButton(formState, modelData)}
          >
            Validate Model
          </Button>
        </div>
      </div>
    </>
  );
}

export default Canvas;
