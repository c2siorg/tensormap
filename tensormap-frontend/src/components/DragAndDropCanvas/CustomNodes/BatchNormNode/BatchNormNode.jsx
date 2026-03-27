import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function BatchNormNode({ id }) {
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-batchnorm px-3 py-1.5 text-xs font-bold text-white">
        BatchNorm
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">Normalizes activations</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

BatchNormNode.propTypes = {
  id: PropTypes.string.isRequired,
};

export default BatchNormNode;
