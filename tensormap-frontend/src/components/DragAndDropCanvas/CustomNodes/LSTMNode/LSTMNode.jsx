import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function LSTMNode({ data, id }) {
  const p = data.params;
  const summary = [
    p.units ? `Units: ${p.units}` : null,
    p.return_sequences === "true" ? "return_seq" : null,
    p.dropout && p.dropout !== "0" ? `Drop: ${p.dropout}` : null,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-recurrent px-3 py-1.5 text-xs font-bold text-white">
        LSTM
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">{summary || "Not configured"}</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}
LSTMNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      units: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      return_sequences: PropTypes.string,
      dropout: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};
export default LSTMNode;
