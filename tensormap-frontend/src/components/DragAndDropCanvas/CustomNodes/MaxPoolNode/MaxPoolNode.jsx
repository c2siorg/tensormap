import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function MaxPoolNode({ data, id }) {
  const { poolX, poolY } = data.params;
  const summary = poolX && poolY ? `Pool: ${poolX}×${poolY}` : "Not configured";

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-maxpool px-3 py-1.5 text-xs font-bold text-white">
        MaxPooling2D
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">{summary}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

MaxPoolNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      poolX: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      poolY: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default MaxPoolNode;
