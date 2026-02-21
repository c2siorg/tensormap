import PropTypes from "prop-types";
import HeatMap from "react-heatmap-grid";

function HeatMapComp({ corrMatrix }) {
  const xLabels = Object.keys(corrMatrix);
  const data_val = xLabels.map((x) => Object.keys(corrMatrix[x]).map((y) => corrMatrix[x][y]));
  return (
    <div>
      <h3 className="mt-5 text-center text-base font-semibold">Co-relation Matrix</h3>
      <div className="flex flex-col items-center justify-center text-[10px] leading-[1.8]">
        <HeatMap
          xLabels={xLabels}
          yLabels={xLabels}
          xLabelWidth={80}
          yLabelWidth={80}
          xLabelsLocation="bottom"
          yLabelsLocation="left"
          data={data_val}
          height={30}
          cellStyle={(background, value, min, max) => ({
            background: `rgba(66, 86, 244, ${1 - (max - value) / (max - min)})`,
            fontSize: "11px",
          })}
          cellRender={(value) => value && `${parseFloat(value).toFixed(4)}`}
        />
      </div>
    </div>
  );
}

HeatMapComp.propTypes = {
  corrMatrix: PropTypes.object.isRequired,
};

export default HeatMapComp;
