import { Handle, Position } from "reactflow";
import PropTypes from "prop-types";

function BatchNormalizationNode({ data }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border">
<<<<<<< HEAD
      <div className="text-sm font-bold">BatchNorm</div>
=======
      <div className="text-sm font-bold">
        {data?.label ?? "BatchNorm"}
      </div>
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)

      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

BatchNormalizationNode.propTypes = {
  data: PropTypes.object,
};

export default BatchNormalizationNode;
