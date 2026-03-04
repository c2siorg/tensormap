import React from "react";
import PropTypes from "prop-types";

function DropoutNode({ data, id }) {
  return (
    <div className="bg-white border rounded shadow p-2">
      <strong>Dropout</strong>
      <div>rate: {data?.params?.rate || 0.5}</div>
    </div>
  );
}

DropoutNode.propTypes = {
  data: PropTypes.object,
  id: PropTypes.string,
};

export default DropoutNode;
