import { Handle, Position } from "reactflow";
import PropTypes from "prop-types";

function DropoutNode({ data }) {
  return (
    <div className="bg-white border rounded shadow px-3 py-2 text-xs">

      <Handle type="target" position={Position.Left} />

      <div className="font-semibold">Dropout</div>
      <div>Rate: {data?.rate ?? 0.5}</div>

      <Handle type="source" position={Position.Right} />

    </div>
  );
}

DropoutNode.propTypes = {
  data: PropTypes.object
};

export default DropoutNode;
