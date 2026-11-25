import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

function TransformationCreator({ features, onAdd }) {
  const transformations = [
    "One Hot Encoding",
    "Categorical to Numerical",
    "Drop Column",
    "Min-Max Normalization",
    "Z-score Standardization",
    "Log Transform",
    "Square Root Transform",
    "Clip Outliers",
    "Drop Missing Rows",
    "Fill Missing Values",
  ];
  const [selectedFeature, setSelectedFeature] = useState(features[0] || "");
  const [selectedTransformation, setSelectedTransformation] = useState(transformations[0]);
  const [fillStrategy, setFillStrategy] = useState("auto");
  const [fillValue, setFillValue] = useState("");
  const [disabledButton, setDisabledButton] = useState(true);

  useEffect(() => {
    if (features.length > 0) {
      if (!selectedFeature || !features.includes(selectedFeature)) {
        setSelectedFeature(features[0]);
      }
    } else {
      setSelectedFeature("");
    }
  }, [features, selectedFeature]);

  useEffect(() => {
    if (selectedFeature && selectedTransformation) setDisabledButton(false);
    else setDisabledButton(true);
  }, [selectedFeature, selectedTransformation]);

  const handleAdd = () => {
    const item = { feature: selectedFeature, transformation: selectedTransformation };
    if (selectedTransformation === "Fill Missing Values") {
      item.params = { strategy: fillStrategy };
      if (fillStrategy === "constant" && fillValue !== "") {
        item.params.fill_value = fillValue;
      }
    }
    onAdd(item);
  };

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="w-[40%]">
        <Select value={selectedFeature} onValueChange={(value) => setSelectedFeature(value)}>
          <SelectTrigger>
            <SelectValue placeholder="Feature" />
          </SelectTrigger>
          <SelectContent>
            {features.map((feature) => (
              <SelectItem key={feature} value={feature}>
                {feature}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="w-[40%]">
        <Select
          value={selectedTransformation}
          onValueChange={(value) => setSelectedTransformation(value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Transformation" />
          </SelectTrigger>
          <SelectContent>
            {transformations.map((transformation) => (
              <SelectItem key={transformation} value={transformation}>
                {transformation}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {selectedTransformation === "Fill Missing Values" && (
        <div className="flex items-center gap-2">
          <div className="w-28">
            <Select value={fillStrategy} onValueChange={(val) => setFillStrategy(val)}>
              <SelectTrigger>
                <SelectValue placeholder="Strategy" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="mean">Mean</SelectItem>
                <SelectItem value="median">Median</SelectItem>
                <SelectItem value="mode">Mode</SelectItem>
                <SelectItem value="constant">Constant</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {fillStrategy === "constant" && (
            <input
              type="text"
              placeholder="Value"
              value={fillValue}
              onChange={(e) => setFillValue(e.target.value)}
              className="h-10 w-24 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background"
            />
          )}
        </div>
      )}
      <Button onClick={handleAdd} disabled={disabledButton}>
        Add Transformation
      </Button>
    </div>
  );
}

TransformationCreator.propTypes = {
  features: PropTypes.arrayOf(PropTypes.string).isRequired,
  onAdd: PropTypes.func.isRequired,
};

export default TransformationCreator;
