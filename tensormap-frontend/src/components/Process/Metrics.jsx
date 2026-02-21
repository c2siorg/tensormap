import PropTypes from "prop-types";
import { Card, CardContent } from "@/components/ui/card";
import HeatMapComp from "./HeatMapComp";
import DataTypes from "./DataTypes";
import MetricTable from "./MetricTable";

function Metrics({ dataTypes, metrics, corrMatrix, fileType }) {
  return fileType === "zip" ? (
    <div className="p-4 text-muted-foreground">Metrics not available for ZIP files</div>
  ) : (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardContent className="p-4">
            <DataTypes dataTypes={dataTypes} />
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardContent className="p-4">
            <MetricTable metrics={metrics} />
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardContent className="p-4">
          <HeatMapComp corrMatrix={corrMatrix} />
        </CardContent>
      </Card>
    </div>
  );
}

Metrics.propTypes = {
  dataTypes: PropTypes.object.isRequired,
  metrics: PropTypes.object.isRequired,
  corrMatrix: PropTypes.object.isRequired,
  fileType: PropTypes.string.isRequired,
};

export default Metrics;
