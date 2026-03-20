import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function AvgPool2DNode({ data, id }) {
  const p = data.params;
  const summary = [
    p.poolX && p.poolY ? `Pool: ${p.poolX}×${p.poolY}` : null,
    p.strideX && p.strideY ? `S: ${p.strideX}×${p.strideY}` : null,
    p.padding ? `P: ${p.padding}` : null,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <div className="w-48 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-pooling px-3 py-1.5 text-xs font-bold text-white">
        AvgPooling2D
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">{summary || "Not configured"}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}
AvgPool2DNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      poolX: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      poolY: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      strideX: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      strideY: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      padding: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};
export default AvgPool2DNode;
