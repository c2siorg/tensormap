import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

// Dynamic color mapping based on the registry category
const getCategoryColor = (category) => {
  switch (category) {
    case "core": return "bg-blue-600";
    case "convolutional": return "bg-green-600";
    case "regularization": return "bg-yellow-500";
    default: return "bg-slate-600";
  }
};

function GenericLayerNode({ data, id }) {
  const { params, registry } = data;
  const displayName = registry?.display_name || "Unknown Layer";
  const categoryColor = getCategoryColor(registry?.category);

  // Dynamically build the summary text from whatever params exist
  const summary = Object.entries(params || {})
    .filter(([, val]) => val !== "" && val !== null && val !== undefined)
    .map(([key, val]) => `${key}: ${val}`)
    .join(", ");

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className={`rounded-t-lg px-3 py-1.5 text-xs font-bold text-white ${categoryColor}`}>
        {displayName}
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">{summary || "Not configured"}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

GenericLayerNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.object.isRequired,
    registry: PropTypes.object,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default GenericLayerNode;