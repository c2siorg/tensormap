import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function EmbeddingNode({ data, id }) {
  const p = data.params;
  const summary = [
    p.input_dim ? `VocabSz: ${p.input_dim}` : null,
    p.output_dim ? `Dim: ${p.output_dim}` : null,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <div className="w-48 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-recurrent px-3 py-1.5 text-xs font-bold text-white">
        Embedding
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">{summary || "Not configured"}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}
EmbeddingNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      input_dim: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      output_dim: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};
export default EmbeddingNode;
