import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";

function DepthwiseConvNode({ data, id }) {
  const { kernelX, kernelY, activation } = data.params;
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-depthwiseconv px-3 py-1.5 text-xs font-bold text-white">
        DepthwiseConv2D
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground space-y-0.5">
        {kernelX !== undefined && kernelX !== "" ? (
          <div>
            Kernel: {kernelX}×{kernelY}
          </div>
        ) : (
          <div>Not configured</div>
        )}
        {activation && <div>Act: {activation}</div>}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

DepthwiseConvNode.propTypes = {
  data: PropTypes.shape({
    params: PropTypes.shape({
      kernelX: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      kernelY: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      activation: PropTypes.string,
    }).isRequired,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default DepthwiseConvNode;
