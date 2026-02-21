import PropTypes from "prop-types";
import { Handle, Position, useStoreApi, useReactFlow } from "reactflow";

import "./InputNode.css";

function InputNode({ data, id }) {
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
              [name]: Number(value),
            },
          };
        }
        return node;
      }),
    );
  };
  return (
    <div className="input-node">
      <div className="node-header">Input Node</div>
      <label htmlFor="dim-1">Dim 1</label>
      <input
        id="dim-1"
        name="dim-1"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params["dim-1"]}
        placeholder="Enter dim-1"
      />
      <br />
      <label htmlFor="dim-2">Dim 2</label>
      <input
        id="dim-2"
        name="dim-2"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params["dim-2"]}
        placeholder="Enter dim-2"
      />
      <br />
      <label htmlFor="dim-3">Dim 3</label>
      <input
        id="dim-3"
        name="dim-3"
        type="number"
        min="0"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params["dim-3"]}
        placeholder="Enter dim-3"
      />
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

InputNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      "dim-1": PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      "dim-2": PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      "dim-3": PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default InputNode;
