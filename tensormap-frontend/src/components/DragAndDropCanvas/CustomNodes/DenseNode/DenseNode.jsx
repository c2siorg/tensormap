import PropTypes from "prop-types";
import { Handle, Position, useReactFlow } from "reactflow";

function DenseNode({ data, id }) {
  const { deleteElements } = useReactFlow();
  const { units, activation } = data.params;
  const summary = [units ? `Units: ${units}` : null, activation ? `Act: ${activation}` : null]
    .filter(Boolean)
    .join(", ");

  const handleDelete = (event) => {
    event.preventDefault();
    event.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="flex items-center justify-between rounded-t-lg bg-node-dense px-3 py-1.5 text-xs font-bold text-white">
        <span>Dense</span>
        <button
          type="button"
          onClick={handleDelete}
          className="rounded px-1 leading-none text-white/80 hover:bg-white/20 hover:text-white"
          aria-label="Delete layer"
          title="Delete layer"
        >
          ×
        </button>
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
