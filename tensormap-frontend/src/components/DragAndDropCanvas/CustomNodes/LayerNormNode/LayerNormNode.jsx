import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function LayerNormNode({ data, id }) {
  const { epsilon } = data.params;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-batchnorm px-3 py-1.5 text-xs font-bold text-white">
        LayerNorm
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {epsilon ? `ε: ${epsilon}` : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}
LayerNormNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      epsilon: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};
export default LayerNormNode;
