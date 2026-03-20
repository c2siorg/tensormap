import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function ReshapeNode({ data, id }) {
  const { target_shape } = data.params;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-flatten px-3 py-1.5 text-xs font-bold text-white">
        Reshape
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {target_shape ? `→ (${target_shape})` : "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}
ReshapeNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      target_shape: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};
export default ReshapeNode;
