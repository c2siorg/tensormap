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
    "Fill Missing Values",
  ];
  const [selectedFeature, setSelectedFeature] = useState(features[0]);
  const [selectedTransformation, setSelectedTransformation] = useState(transformations[0]);
  const [disabledButton, setDisabledButton] = useState(true);

  useEffect(() => {
    if (selectedFeature && selectedTransformation) setDisabledButton(false);
    else setDisabledButton(true);
  }, [selectedFeature, selectedTransformation]);

  const handleAdd = () => {
    onAdd({ feature: selectedFeature, transformation: selectedTransformation });
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
