import { Handle, Position, useStoreApi, useReactFlow } from 'reactflow';
import './DropoutNode.css';

const DropoutNode = ({ data, id }) => {
  const { setNodes } = useReactFlow();
  const store = useStoreApi();

  const updateNodeState = (evt) => {
    const { nodeInternals } = store.getState();
    const { value } = evt.target;
    setNodes(
      Array.from(nodeInternals.values()).map((node) => {
        if (node.id === id) {
          node.data = {
            ...node.data,
            params: {
              ...node.data.params,
              dropoutRate: Number(value),
            },
          };
        }
        return node;
      })
    );
  };

  return (
    <div className="dropout-node">
      <Handle type="target" position={Position.Left} isConnectable={true} id={`${id}_in`} />
      <div className="node-header">Dropout Node</div>
      <label htmlFor="dropout-rate">Dropout Rate</label>
      <input
        id="dropout-rate"
        name="dropout-rate"
        type="number"
        min="0"
        max="1"
        step="0.1"
        onChange={updateNodeState}
        className="nodrag"
        value={data.params["dropoutRate"]}
        placeholder="Enter the dropout rate"
      />
      <Handle type="source" position={Position.Right} isConnectable={true} id={`${id}_out`} />
    </div>
  );
};

export default DropoutNode;
