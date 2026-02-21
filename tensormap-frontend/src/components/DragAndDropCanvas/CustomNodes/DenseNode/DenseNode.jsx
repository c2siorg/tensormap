import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function DenseNode({ data, id }) {
  const { units, activation } = data.params;
  const summary = [units ? `Units: ${units}` : null, activation ? `Act: ${activation}` : null]
    .filter(Boolean)
    .join(", ");

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-dense px-3 py-1.5 text-xs font-bold text-white">
        Dense
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">{summary || "Not configured"}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

DenseNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      units: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      activation: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default DenseNode;
