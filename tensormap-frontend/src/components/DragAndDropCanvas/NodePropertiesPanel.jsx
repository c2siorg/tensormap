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

const denseActivations = [
  { value: "relu", label: "ReLU" },
  { value: "linear", label: "Linear" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "softmax", label: "Softmax" },
  { value: "tanh", label: "Tanh" },
];

const convActivations = [
  { value: "none", label: "None" },
  { value: "relu", label: "ReLU" },
];

const convPaddings = [
  { value: "valid", label: "Valid" },
  { value: "same", label: "Same" },
];

function NodePropertiesPanel({
  selectedNode,
  modelName,
  onModelNameChange,
  onSave,
  canSave,
  onNodeUpdate,
}) {
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
            Validate &amp; Save
          </Button>
        </CardContent>
      </Card>
    );
  }

  const { type, data, id } = selectedNode;
  const params = data.params;

  const updateParam = (name, value) => {
    onNodeUpdate(id, { ...params, [name]: value });
  };

  if (type === "custominput") {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Input Layer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Dim 1</Label>
            <Input
              type="number"
              min="0"
              placeholder="Dimension 1"
              value={params["dim-1"]}
              onChange={(e) => updateParam("dim-1", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Dim 2</Label>
            <Input
              type="number"
              min="0"
              placeholder="Dimension 2 (optional)"
              value={params["dim-2"]}
              onChange={(e) => updateParam("dim-2", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Dim 3</Label>
            <Input
              type="number"
              min="0"
              placeholder="Dimension 3 (optional)"
              value={params["dim-3"]}
              onChange={(e) => updateParam("dim-3", Number(e.target.value))}
            />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (type === "customdense") {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Dense Layer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Units</Label>
            <Input
              type="number"
              min="1"
              placeholder="Number of units"
              value={params.units}
              onChange={(e) => updateParam("units", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Activation</Label>
            <Select value={params.activation} onValueChange={(v) => updateParam("activation", v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select activation" />
              </SelectTrigger>
              <SelectContent>
                {denseActivations.map((a) => (
                  <SelectItem key={a.value} value={a.value}>
                    {a.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (type === "customflatten") {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Flatten Layer</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No configurable parameters</p>
        </CardContent>
      </Card>
    );
  }

  if (type === "customconv") {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Conv2D Layer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Filters</Label>
            <Input
              type="number"
              min="1"
              placeholder="Filter count"
              value={params.filter}
              onChange={(e) => updateParam("filter", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Kernel X</Label>
            <Input
              type="number"
              min="1"
              placeholder="Kernel X"
              value={params.kernelX}
              onChange={(e) => updateParam("kernelX", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Kernel Y</Label>
            <Input
              type="number"
              min="1"
              placeholder="Kernel Y"
              value={params.kernelY}
              onChange={(e) => updateParam("kernelY", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Stride X</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride X"
              value={params.strideX}
              onChange={(e) => updateParam("strideX", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Stride Y</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride Y"
              value={params.strideY}
              onChange={(e) => updateParam("strideY", Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label>Padding</Label>
            <Select value={params.padding} onValueChange={(v) => updateParam("padding", v)}>
              <SelectTrigger>
                <SelectValue placeholder="Padding" />
              </SelectTrigger>
              <SelectContent>
                {convPaddings.map((p) => (
                  <SelectItem key={p.value} value={p.value}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label>Activation</Label>
            <Select value={params.activation} onValueChange={(v) => updateParam("activation", v)}>
              <SelectTrigger>
                <SelectValue placeholder="Activation" />
              </SelectTrigger>
              <SelectContent>
                {convActivations.map((a) => (
                  <SelectItem key={a.value} value={a.value}>
                    {a.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
    );
  }

  return null;
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
