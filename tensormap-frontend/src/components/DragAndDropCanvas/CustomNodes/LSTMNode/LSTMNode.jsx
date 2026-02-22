import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function LSTMNode({ data, id }) {
  const { units, returnSequences } = data.params;
  const parsedUnits = Number(units);
  const configured = units !== "" && units !== undefined && !isNaN(parsedUnits) && parsedUnits > 0;

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-lstm px-3 py-1.5 text-xs font-bold text-white">LSTM</div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {configured
          ? `Units: ${units}${returnSequences === "true" ? " \u00b7 seq" : ""}`
          : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

LSTMNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      units: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      returnSequences: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default LSTMNode;
