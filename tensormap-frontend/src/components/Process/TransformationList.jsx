import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";

function TransformationList({ transformations, onDelete }) {
  return (
    <div className="my-4 overflow-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b">
            <th className="border px-3 py-2 text-left font-semibold">Feature</th>
            <th className="border px-3 py-2 text-left font-semibold">Transformation</th>
            <th className="border px-3 py-2 text-center font-semibold">Delete Transformation</th>
          </tr>
        </thead>
        <tbody>
          {transformations.map((transformation, index) => (
            <tr key={index} className="border-b">
              <td className="border px-3 py-2">{transformation.feature}</td>
              <td className="border px-3 py-2">{transformation.transformation}</td>
              <td className="border px-3 py-2 text-center">
                <Button variant="destructive" size="sm" onClick={() => onDelete(index)}>
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

TransformationList.propTypes = {
  transformations: PropTypes.arrayOf(
    PropTypes.shape({
      feature: PropTypes.string.isRequired,
      transformation: PropTypes.string.isRequired,
    }),
  ).isRequired,
  onDelete: PropTypes.func.isRequired,
};

export default TransformationList;
