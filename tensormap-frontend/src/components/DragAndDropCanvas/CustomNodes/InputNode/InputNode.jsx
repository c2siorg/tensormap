import PropTypes from "prop-types";
import { Handle, Position, useReactFlow } from "reactflow";

function InputNode({ data, id }) {
  const { deleteElements } = useReactFlow();
  const dims = [data.params["dim-1"], data.params["dim-2"], data.params["dim-3"]]
    .filter((d) => d !== "" && d !== undefined)
    .join(" × ");

  const handleDelete = (event) => {
    event.preventDefault();
    event.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <div className="flex items-center justify-between rounded-t-lg bg-node-input px-3 py-1.5 text-xs font-bold text-white">
        <span>Input</span>
        <button
          type="button"
          onClick={handleDelete}
          className="rounded px-1 leading-none text-white/80 hover:bg-white/20 hover:text-white"
          aria-label="Delete layer"
          title="Delete layer"
          data-testid="input-node-delete-button"
        >
          ×
        </button>
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {dims ? `Dim: ${dims}` : "No dimensions set"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

InputNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      "dim-1": PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      "dim-2": PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      "dim-3": PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default InputNode;
