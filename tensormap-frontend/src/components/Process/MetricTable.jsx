import PropTypes from "prop-types";

function MetricTable({ metrics }) {
  const roundValue = (value) => (typeof value === "number" ? value.toFixed(2) : value);

  return (
    <div>
      <h3 className="mt-5 text-center text-base font-semibold">Dataset Metrics</h3>
      <div className="overflow-auto">
        <table className="w-full border-collapse text-[0.7em]">
          <thead>
            <tr className="border-b">
              <th className="border px-2 py-1.5 text-left font-semibold">Feature</th>
              <th className="border px-2 py-1.5 text-left font-semibold">Count</th>
              <th className="border px-2 py-1.5 text-left font-semibold">Mean</th>
              <th className="border px-2 py-1.5 text-left font-semibold">Standard Deviation</th>
              <th className="border px-2 py-1.5 text-left font-semibold">Min</th>
              <th className="border px-2 py-1.5 text-left font-semibold">25%</th>
              <th className="border px-2 py-1.5 text-left font-semibold">50%</th>
              <th className="border px-2 py-1.5 text-left font-semibold">75%</th>
              <th className="border px-2 py-1.5 text-left font-semibold">Max</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([feature, values]) => (
              <tr key={feature} className="border-b">
                <td className="border px-2 py-1.5">{roundValue(feature)}</td>
                <td className="border px-2 py-1.5">{roundValue(values.count)}</td>
                <td className="border px-2 py-1.5">{roundValue(values.mean)}</td>
                <td className="border px-2 py-1.5">{roundValue(values.std)}</td>
                <td className="border px-2 py-1.5">{roundValue(values.min)}</td>
                <td className="border px-2 py-1.5">{roundValue(values["25%"])}</td>
                <td className="border px-2 py-1.5">{roundValue(values["50%"])}</td>
                <td className="border px-2 py-1.5">{roundValue(values["75%"])}</td>
                <td className="border px-2 py-1.5">{roundValue(values.max)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

MetricTable.propTypes = {
  metrics: PropTypes.object.isRequired,
};

export default MetricTable;
