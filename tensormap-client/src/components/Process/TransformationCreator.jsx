import { useEffect, useState } from "react";
import { Dropdown, Button } from "semantic-ui-react";

function TransformationCreator({ features, onAdd }) {
    const transformations = ["One Hot Encode", "Convert Class Data"];
    const [selectedFeature, setSelectedFeature] = useState(features[0]);
    const [selectedTransformation, setSelectedTransformation] = useState(transformations[0]);

    useEffect(() => {
        setSelectedFeature(selectedFeature);
    }, [selectedFeature]);

    const handleFeatureChange = (event, val) => {
        setSelectedFeature(val.value);
    };

    const handleTransformationChange = (event, val) => {
        setSelectedTransformation(val.value);
    };

    const handleAdd = () => {
        onAdd({ feature: selectedFeature, transformation: selectedTransformation });
    };

    return (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Dropdown
                style={{ width: "40%" }}
                placeholder="Feature"
                search
                selection
                onChange={handleFeatureChange}
                options={features.map((feature) => ({ key: feature, text: feature, value: feature }))}
            />
            <Dropdown
                style={{ width: "40%" }}
                placeholder="Transformation"
                search
                selection
                onChange={handleTransformationChange}
                options={transformations.map((tranformation) => ({ key: tranformation, text: tranformation, value: tranformation }))}
            />
            <Button color="green" size="large" onClick={handleAdd}>
                Add Transformation
            </Button>
        </div>
    );
}

export default TransformationCreator;