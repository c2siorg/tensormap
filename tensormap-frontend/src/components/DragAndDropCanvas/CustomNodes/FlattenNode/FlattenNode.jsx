import PropTypes from "prop-types";
import { Handle, Position, useReactFlow } from "reactflow";

function FlattenNode({ id }) {
  const { deleteElements } = useReactFlow();

  const handleDelete = (event) => {
    event.preventDefault();
    event.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="flex items-center justify-between rounded-t-lg bg-node-flatten px-3 py-1.5 text-xs font-bold text-white">
        <span>Flatten</span>
        <button
          type="button"
          onClick={handleDelete}
          className="rounded px-1 leading-none text-white/80 hover:bg-white/20 hover:text-white"
          aria-label="Delete layer"
          title="Delete layer"
          data-testid="flatten-node-delete-button"
        >
          ×
        </button>
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
