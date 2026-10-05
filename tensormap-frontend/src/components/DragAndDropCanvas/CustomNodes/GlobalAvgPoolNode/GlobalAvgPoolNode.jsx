import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";
import NodeShapeBadge from "../../../nodes/NodeShapeBadge";

function GlobalAvgPoolNode({ data, id }) {
  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className="rounded-t-lg bg-node-globalavgpool px-3 py-1.5 text-xs font-bold text-white">
        GlobalAvgPool2D
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">No parameters</div>
      <NodeShapeBadge analysis={data?.analysis} />
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

GlobalAvgPoolNode.propTypes = {
  data: PropTypes.shape({ analysis: PropTypes.object }),
  id: PropTypes.string.isRequired,
};

export default GlobalAvgPoolNode;
