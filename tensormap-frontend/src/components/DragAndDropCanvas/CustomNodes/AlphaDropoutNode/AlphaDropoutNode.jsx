import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function AlphaDropoutNode({ data, id }) {
  const rate = data.params?.rate ?? "";
  const configured =
    String(rate).trim() !== "" && !isNaN(Number(rate)) && Number(rate) > 0 && Number(rate) < 1;

  return (
    <div className="w-56 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-alphadropout px-3 py-1.5 text-xs font-bold text-white">
        AlphaDropout
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {configured ? `Rate: ${rate}` : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

AlphaDropoutNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      rate: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default AlphaDropoutNode;
