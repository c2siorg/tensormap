import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";
import NodeDeleteButton from "../shared/NodeDeleteButton";

function FlattenNode({ id }) {
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="flex items-center justify-between rounded-t-lg bg-node-flatten px-3 py-1.5 text-xs font-bold text-white">
        <span>Flatten</span>
        <NodeDeleteButton id={id} testId="flatten-node-delete-button" />
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
