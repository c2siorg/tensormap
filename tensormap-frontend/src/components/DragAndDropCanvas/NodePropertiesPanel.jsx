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

const ACTIVATIONS = [
  { value: "none", label: "None" },
  { value: "relu", label: "ReLU" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "tanh", label: "Tanh" },
  { value: "softmax", label: "Softmax" },
  { value: "elu", label: "ELU" },
  { value: "selu", label: "SELU" },
];

const convPaddings = [
  { value: "valid", label: "Valid" },
  { value: "same", label: "Same" },
];

/**
 * Update a parameter, handling empty string as null to prevent coercion to 0.
 */
const updateParam = (params, onNodeUpdate, id) => (name, value) => {
  if (value === "" || value === null || value === undefined) {
    onNodeUpdate(id, { ...params, [name]: null });
  } else {
    onNodeUpdate(id, { ...params, [name]: value });
  }
};

/**
 * Handle number input changes, converting empty string to null.
 */
const handleNumberChange = (params, onNodeUpdate, id, name) => (e) => {
  const val = e.target.value;
  const update = updateParam(params, onNodeUpdate, id);
  if (val === "") {
    update(name, null);
  } else {
    const num = Number(val);
    if (!isNaN(num)) {
      update(name, num);
    }
  }
};

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
            Validate & Save
          </Button>
        </CardContent>
      </Card>
    );
  }

  const { type, data, id } = selectedNode;
  const params = data.params;
  const handleChange = handleNumberChange(params, onNodeUpdate, id);
  const doUpdate = updateParam(params, onNodeUpdate, id);

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
              min="1"
              step="1"
              placeholder="Dimension 1"
              value={params["dim-1"] ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                if (val === "") {
                  doUpdate("dim-1", null);
                } else {
                  doUpdate("dim-1", Math.max(1, Math.floor(Number(val))));
                }
              }}
            />
          </div>
          <div className="space-y-1">
            <Label>Dim 2</Label>
            <Input
              type="number"
              min="1"
              step="1"
              placeholder="Dimension 2 (optional)"
              value={params["dim-2"] ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                if (val === "") {
                  doUpdate("dim-2", null);
                } else {
                  doUpdate("dim-2", Math.max(1, Math.floor(Number(val))));
                }
              }}
            />
          </div>
          <div className="space-y-1">
            <Label>Dim 3</Label>
            <Input
              type="number"
              min="1"
              step="1"
              placeholder="Dimension 3 (optional)"
              value={params["dim-3"] ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                if (val === "") {
                  doUpdate("dim-3", null);
                } else {
                  doUpdate("dim-3", Math.max(1, Math.floor(Number(val))));
                }
              }}
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
              value={params.units ?? ""}
              onChange={handleChange("units")}
            />
          </div>
          <div className="space-y-1">
            <Label>Activation</Label>
            <Select
              value={params.activation ?? ""}
              onValueChange={(v) => doUpdate("activation", v || null)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select activation" />
              </SelectTrigger>
              <SelectContent>
                {ACTIVATIONS.map((a) => (
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
              value={params.filter ?? ""}
              onChange={handleChange("filter")}
            />
          </div>
          <div className="space-y-1">
            <Label>Kernel X</Label>
            <Input
              type="number"
              min="1"
              placeholder="Kernel X"
              value={params.kernelX ?? ""}
              onChange={handleChange("kernelX")}
            />
          </div>
          <div className="space-y-1">
            <Label>Kernel Y</Label>
            <Input
              type="number"
              min="1"
              placeholder="Kernel Y"
              value={params.kernelY ?? ""}
              onChange={handleChange("kernelY")}
            />
          </div>
          <div className="space-y-1">
            <Label>Stride X</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride X"
              value={params.strideX ?? ""}
              onChange={handleChange("strideX")}
            />
          </div>
          <div className="space-y-1">
            <Label>Stride Y</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride Y"
              value={params.strideY ?? ""}
              onChange={handleChange("strideY")}
            />
          </div>
          <div className="space-y-1">
            <Label>Padding</Label>
            <Select
              value={params.padding ?? ""}
              onValueChange={(v) => doUpdate("padding", v || null)}
            >
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
            <Select
              value={params.activation ?? ""}
              onValueChange={(v) => doUpdate("activation", v || null)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Activation" />
              </SelectTrigger>
              <SelectContent>
                {ACTIVATIONS.map((a) => (
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

  if (type === "customdropout") {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Dropout</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Rate (0–0.9)</Label>
            <Input
              type="number"
              min="0"
              max="0.9"
              step="0.1"
              value={params.rate ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                if (val === "") {
                  doUpdate("rate", null);
                } else {
                  const v = parseFloat(val);
                  if (!isNaN(v)) {
                    doUpdate("rate", Math.min(0.9, Math.max(0, v)));
                  }
                }
              }}
            />
          </div>
        </CardContent>
      </Card>
    );
  }
  if (type === "custommaxpool") {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">MaxPooling2D Layer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Pool Size</Label>
            <Input
              type="number"
              min="1"
              placeholder="Pool size"
              value={params.pool_size ?? ""}
              onChange={handleChange("pool_size")}
            />
          </div>
          <div className="space-y-1">
            <Label>Stride</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride"
              value={params.stride ?? ""}
              onChange={handleChange("stride")}
            />
          </div>
          <div className="space-y-1">
            <Label>Padding</Label>
            <Select
              value={params.padding ?? ""}
              onValueChange={(v) => doUpdate("padding", v || null)}
            >
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
