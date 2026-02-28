import PropTypes from "prop-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

function NodePropertiesPanel({
  selectedNode,
  modelName,
  onModelNameChange,
  onSave,
  canSave,
  onNodeUpdate,
}) {
  // If no node is selected, show the default Save Model panel
  if (!selectedNode) {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Save Model</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Model Name</Label>
            <Input
              placeholder="Enter model name"
              value={modelName}
              onChange={(e) => onModelNameChange(e.target.value)}
            />
          </div>
          <Button className="w-full" onClick={onSave} disabled={!canSave}>
            Validate & Save
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Extract the dynamically generated data from our new GenericLayerNode
  const { id, data } = selectedNode;
  const params = data.params || {};
  const registry = data.registry || {};

  // Safely update parameters, casting to number if required by the API
  const updateParam = (name, value, expectedType) => {
    let parsedValue = value;
    if (expectedType === "int" || expectedType === "float") {
      parsedValue = value === "" ? "" : Number(value);
    }
    onNodeUpdate(id, { ...params, [name]: parsedValue });
  };

  // Helper to make parameter keys look nice (e.g., 'dim-1' -> 'Dim 1')
  const formatLabel = (str) => {
    const spaced = str.replace(/-/g, " ").replace(/([A-Z])/g, " $1");
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  };

  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="text-sm">{registry.display_name || "Layer"} Properties</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* If the layer has no parameters (like Flatten), show a fallback message */}
        {!registry.params || Object.keys(registry.params).length === 0 ? (
          <p className="text-sm text-muted-foreground">No configurable parameters</p>
        ) : (
          /* Loop through the JSON API parameters to build the UI dynamically */
          Object.entries(registry.params).map(([paramKey, paramConfig]) => {
            const isSelect = Array.isArray(paramConfig.options);
            const currentValue = params[paramKey] !== undefined ? params[paramKey] : "";

            return (
              <div key={paramKey} className="space-y-1">
                <Label>{formatLabel(paramKey)}</Label>
                
                {isSelect ? (
                  <Select
                    value={String(currentValue)}
                    onValueChange={(v) => updateParam(paramKey, v, paramConfig.type)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={`Select ${formatLabel(paramKey)}`} />
                    </SelectTrigger>
                    <SelectContent>
                      {paramConfig.options.map((opt) => (
                        <SelectItem key={opt} value={opt}>
                          {opt}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    type={paramConfig.type === "int" || paramConfig.type === "float" ? "number" : "text"}
                    placeholder={`Enter ${formatLabel(paramKey)}`}
                    value={currentValue}
                    onChange={(e) => updateParam(paramKey, e.target.value, paramConfig.type)}
                  />
                )}
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}

NodePropertiesPanel.propTypes = {
  selectedNode: PropTypes.object,
  modelName: PropTypes.string.isRequired,
  onModelNameChange: PropTypes.func.isRequired,
  onSave: PropTypes.func.isRequired,
  canSave: PropTypes.bool.isRequired,
  onNodeUpdate: PropTypes.func.isRequired,
};

export default NodePropertiesPanel;