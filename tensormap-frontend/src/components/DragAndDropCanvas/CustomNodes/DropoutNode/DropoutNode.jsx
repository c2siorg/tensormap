import { Handle, Position } from "reactflow";
import PropTypes from "prop-types";

function DropoutNode({ data, id }) {
  return (
    <div className="bg-white border rounded shadow px-3 py-2 text-xs">
      <Handle type="target" position={Position.Left} id="input" />

      <div className="font-semibold">Dropout</div>

      <label className="flex items-center gap-1">
        Rate:
        <input
          type="number"
          min={0}
          max={0.99}
          step={0.01}
          defaultValue={data?.rate ?? 0.5}
          onChange={(e) => {
            const newRate = parseFloat(e.target.value);
            if (!isNaN(newRate) && newRate >= 0 && newRate < 1) {
              data?.onChange?.(id, { rate: newRate });
            }
          }}
          className="w-14 border rounded px-1"
        />
      </label>

      <Handle type="source" position={Position.Right} id="output" />
    </div>
  );
}

DropoutNode.propTypes = {
  id: PropTypes.string.isRequired,
  selected: PropTypes.bool,
  data: PropTypes.shape({
    rate: PropTypes.number,
    onChange: PropTypes.func,
  }),
};

export default DropoutNode;
