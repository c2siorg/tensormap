import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function LSTMNode({ data, id }) {
  const { units, returnSequences, activation, recurrentActivation } = data.params;
  const parsedUnits = Number(units);
  const configured = String(units).trim() !== "" && parsedUnits > 0;
  const activationLabel = activation || "tanh";
  const recurrentLabel = recurrentActivation || "sigmoid";

  return (
    <div className="w-56 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-lstm px-3 py-1.5 text-xs font-bold text-white">LSTM</div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {configured
          ? `Units: ${units}${returnSequences === "true" || returnSequences === true ? " • seq" : ""}`
          : "Not configured"}
      </div>
      {configured && (
        <div className="border-t px-3 py-1.5 text-xs text-gray-500">
          <div>
            {activationLabel} / {recurrentLabel}
          </div>
        </div>
      )}
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

LSTMNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      units: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      returnSequences: PropTypes.string,
      activation: PropTypes.string,
      recurrentActivation: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default LSTMNode;
