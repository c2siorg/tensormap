import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function FlattenNode({ id }) {
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-flatten px-3 py-1.5 text-xs font-bold text-white">
        Flatten
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">No parameters</div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

FlattenNode.propTypes = {
  id: PropTypes.string.isRequired,
};

export default FlattenNode;
