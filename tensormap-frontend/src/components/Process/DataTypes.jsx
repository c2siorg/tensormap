import PropTypes from "prop-types";

function DataTypes({ dataTypes }) {
  return (
    <div>
      <h3 className="mt-5 text-center text-base font-semibold">Datatypes</h3>
      <table className="w-full text-[0.8em]">
        <thead>
          <tr className="border-b">
            <th className="px-3 py-2 text-left font-semibold">Column Name</th>
            <th className="px-3 py-2 text-left font-semibold">Data Type</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(dataTypes).map(([column, dtype]) => (
            <tr key={column} className="border-b">
              <td className="px-3 py-2">{column}</td>
              <td className="px-3 py-2">{dtype}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

DataTypes.propTypes = {
  dataTypes: PropTypes.object.isRequired,
};

export default DataTypes;
