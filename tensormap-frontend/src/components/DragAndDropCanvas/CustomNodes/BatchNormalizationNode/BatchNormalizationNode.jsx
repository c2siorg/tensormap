import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function BatchNormalizationNode({ data, id }) {
  const { momentum, epsilon } = data.params;
  const isConfigured = momentum !== "" && momentum !== undefined;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-batchnorm px-3 py-1.5 text-xs font-bold text-white">
        BatchNorm
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {isConfigured ? `Momentum: ${momentum} | Epsilon: ${epsilon}` : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

BatchNormalizationNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      momentum: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      epsilon: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default BatchNormalizationNode;
