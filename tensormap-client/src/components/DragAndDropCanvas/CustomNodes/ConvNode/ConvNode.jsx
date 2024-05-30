import { Handle, Position, useStoreApi, useReactFlow } from "reactflow";
import "./ConvNode.css";

function ConvNode({ data, id }) {
  const paddingOptions = ["valid", "same"];
  const activationOptions = ["none", "relu"];

  const { setNodes } = useReactFlow();
  const store = useStoreApi();
  const updateNodeState = (evt) => {
    const { nodeInternals } = store.getState();
    const { name, value } = evt.target;
    setNodes(
      Array.from(nodeInternals.values()).map((node) => {
        if (node.id === id) {
          node.data = {
            ...node.data,
            params: {
              ...node.data.params,
              [name]:
                name === "filter" || name === "strideX" || name === "strideY"
                  ? Number(value)
                  : value,
            },
          };
        }
        console.log(node);
        return node;
      }),
    );
  };
  return (
    <div className="conv-node">
      <Handle
        type="target"
        position={Position.Left}
        isConnectable
        id={`${id}_in`}
      />
      <div className="node-header">Conv Node</div>

      <label htmlFor="filter">Filter</label>
      <input
        id="filter"
        name="filter"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params.filter}
        placeholder="Enter the filter size"
      />

      <label htmlFor="padding">Padding</label>
      <select
        id="padding"
        className="padding"
        name="padding"
        onChange={updateNodeState}
        value={data.params.padding}
      >
        {paddingOptions.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>

      <label htmlFor="activation">Activation</label>
      <select
        id="activation"
        className="activation"
        name="activation"
        onChange={updateNodeState}
        value={data.params.activation}
      >
        {activationOptions.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>

      <label htmlFor="kernelX">Kernel X</label>
      <input
        id="kernelX"
        name="kernelX"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params.kernelX}
        placeholder="Enter the kernel X size"
      />

      <label htmlFor="kernelY">Kernel Y</label>
      <input
        id="kernelY"
        name="kernelY"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params.kernelY}
        placeholder="Enter the kernel Y size"
      />

      <label htmlFor="strideX">Stride X</label>
      <input
        id="strideX"
        name="strideX"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params.strideX}
        placeholder="Enter the stride X"
      />

      <label htmlFor="strideY">Stride Y</label>
      <input
        id="strideY"
        name="strideY"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params.strideY}
        placeholder="Enter the stride Y"
      />

      <Handle
        type="source"
        position={Position.Right}
        isConnectable
        id={`${id}_out`}
      />
    </div>
  );
}

export default ConvNode;
