import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function DropoutNode({ data, id }) {
  const { rate } = data.params;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-dropout px-3 py-1.5 text-xs font-bold text-white">
        Dropout
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {rate !== "" && rate !== undefined ? `Rate: ${rate}` : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

DropoutNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      rate: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default DropoutNode;
