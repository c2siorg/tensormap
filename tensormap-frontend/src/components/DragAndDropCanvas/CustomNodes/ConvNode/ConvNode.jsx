import PropTypes from "prop-types";
import { Handle, Position, useReactFlow } from "reactflow";

function ConvNode({ data, id }) {
  const { deleteElements } = useReactFlow();
  const p = data.params;
  const parts = [
    p.filter ? `F: ${p.filter}` : null,
    p.kernelX && p.kernelY ? `K: ${p.kernelX}×${p.kernelY}` : null,
    p.strideX && p.strideY ? `S: ${p.strideX}×${p.strideY}` : null,
    p.padding ? `P: ${p.padding}` : null,
    p.activation && p.activation !== "none" ? `Act: ${p.activation}` : null,
  ]
    .filter(Boolean)
    .join(", ");

  const handleDelete = (event) => {
    event.preventDefault();
    event.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <div className="w-48 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="flex items-center justify-between rounded-t-lg bg-node-conv px-3 py-1.5 text-xs font-bold text-white">
        <span>Conv2D</span>
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
      <div className="px-3 py-2 text-xs text-muted-foreground">{parts || "Not configured"}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

ConvNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      filter: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      padding: PropTypes.string,
      activation: PropTypes.string,
      kernelX: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      kernelY: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      strideX: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      strideY: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default ConvNode;
