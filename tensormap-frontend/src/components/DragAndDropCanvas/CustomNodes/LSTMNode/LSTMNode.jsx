import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function LSTMNode({ data, id }) {
  const { units, return_sequences, activation } = data.params;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-lstm px-3 py-1.5 text-xs font-bold text-white">
        LSTM
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground space-y-0.5">
        {units !== undefined && units !== "" ? <div>Units: {units}</div> : <div>Not configured</div>}
        {activation && <div>Act: {activation}</div>}
        {return_sequences === "true" && <div>Return seq: yes</div>}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

LSTMNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      units: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      return_sequences: PropTypes.string,
      activation: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default LSTMNode;
