import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function MaxPoolingNode({ data, id }) {
  const { pool_size, stride, padding } = data.params;
  const isConfigured = pool_size !== "" && pool_size !== undefined;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-conv px-3 py-1.5 text-xs font-bold text-white">
        MaxPooling2D
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {isConfigured ? `Pool: ${pool_size} | Stride: ${stride} | ${padding}` : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

MaxPoolingNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      pool_size: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      stride: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      padding: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default MaxPoolingNode;
