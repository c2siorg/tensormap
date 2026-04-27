import PropTypes from "prop-types";
import { useReactFlow } from "reactflow";

function NodeDeleteButton({ id, testId }) {
  const { deleteElements } = useReactFlow();

  const handleDelete = (event) => {
    event.preventDefault();
    event.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <button
      type="button"
      onClick={handleDelete}
      className="rounded px-1 leading-none text-white/80 hover:bg-white/20 hover:text-white"
      aria-label="Delete layer"
      title="Delete layer"
      data-testid={testId}
    >
      ×
    </button>
  );
}

NodeDeleteButton.propTypes = {
  id: PropTypes.string.isRequired,
  testId: PropTypes.string.isRequired,
};

export default NodeDeleteButton;
