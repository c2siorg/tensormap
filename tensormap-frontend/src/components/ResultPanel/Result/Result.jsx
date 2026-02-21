import PropTypes from "prop-types";

function Result(props) {
  return (
    <div className="mx-2.5 mt-2.5 rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-800">
      {props.result}
    </div>
  );
}

Result.propTypes = {
  result: PropTypes.string,
};

export default Result;
