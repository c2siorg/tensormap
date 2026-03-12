import { Handle, Position } from "reactflow";
import PropTypes from "prop-types";

function BatchNormalizationNode({ data }) {
  return (
    <div className="px-4 py-2 shadow-md rounded-md bg-white border">
      <div className="text-sm font-bold">
        {data?.label ?? "BatchNorm"}
      </div>

      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

BatchNormalizationNode.propTypes = {
  data: PropTypes.object,
};

export default BatchNormalizationNode;
