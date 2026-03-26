import PropTypes from "prop-types";
import { X } from "lucide-react";
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

function NodeConfigPanel({ isOpen, onClose, selectedNode, onNodeUpdate }) {
  if (!isOpen || !selectedNode) return null;

  const { type, data, id } = selectedNode;
  const params = data.params;

  const updateParam = (name, value) => {
    onNodeUpdate(id, { ...params, [name]: value });
  };

  const renderContent = () => {
    if (type === "custominput") {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Dim 1</Label>
            <Input
              type="number"
              min="0"
              placeholder="Dimension 1"
              value={params["dim-1"] ?? ""}
              onChange={(e) =>
                updateParam("dim-1", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Dim 2</Label>
            <Input
              type="number"
              min="0"
              placeholder="Dimension 2 (optional)"
              value={params["dim-2"] ?? ""}
              onChange={(e) =>
                updateParam("dim-2", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Dim 3</Label>
            <Input
              type="number"
              min="0"
              placeholder="Dimension 3 (optional)"
              value={params["dim-3"] ?? ""}
              onChange={(e) =>
                updateParam("dim-3", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
        </div>
      );
    }

    if (type === "customdense") {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Units</Label>
            <Input
              type="number"
              min="1"
              placeholder="Number of units"
              value={params.units ?? ""}
              onChange={(e) =>
                updateParam("units", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Activation</Label>
            <Select value={params.activation} onValueChange={(v) => updateParam("activation", v)}>
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
        </div>
      );
    }

    if (type === "customflatten") {
      return (
        <p className="text-sm text-muted-foreground">
          No configurable parameters for Flatten layer
        </p>
      );
    }

    if (type === "customconv") {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Filters</Label>
            <Input
              type="number"
              min="1"
              placeholder="Filter count"
              value={params.filter ?? ""}
              onChange={(e) =>
                updateParam("filter", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Kernel X</Label>
            <Input
              type="number"
              min="1"
              placeholder="Kernel X"
              value={params.kernelX ?? ""}
              onChange={(e) =>
                updateParam("kernelX", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Kernel Y</Label>
            <Input
              type="number"
              min="1"
              placeholder="Kernel Y"
              value={params.kernelY ?? ""}
              onChange={(e) =>
                updateParam("kernelY", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Stride X</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride X"
              value={params.strideX ?? ""}
              onChange={(e) =>
                updateParam("strideX", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
            <Label>Stride Y</Label>
            <Input
              type="number"
              min="1"
              placeholder="Stride Y"
              value={params.strideY ?? ""}
              onChange={(e) =>
                updateParam("strideY", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
          <div className="space-y-2">
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
          <div className="space-y-2">
            <Label>Activation</Label>
            <Select value={params.activation} onValueChange={(v) => updateParam("activation", v)}>
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
        </div>
      );
    }

    if (type === "customdropout") {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Rate (0–1)</Label>
            <Input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={params.rate ?? ""}
              onChange={(e) =>
                updateParam("rate", e.target.value === "" ? "" : Number(e.target.value))
              }
            />
          </div>
        </div>
      );
    }

    return null;
  };

  const getLayerTitle = () => {
    switch (type) {
      case "custominput":
        return "Input Layer";
      case "customdense":
        return "Dense Layer";
      case "customflatten":
        return "Flatten Layer";
      case "customconv":
        return "Conv2D Layer";
      case "customdropout":
        return "Dropout";
      default:
        return "Layer Configuration";
    }
  };

  return (
    <>
      {/* Overlay to catch clicks outside the panel and close it */}
      {isOpen && <div className="fixed inset-0 z-40" onClick={onClose} aria-hidden="true" />}

      {/* Slide-in panel */}
      <div
        className={`fixed inset-y-0 right-0 z-50 w-80 transform bg-background border-l shadow-2xl transition-transform duration-300 ease-in-out ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="flex items-center justify-between border-b px-4 py-3">
            <h2 className="text-lg font-semibold">{getLayerTitle()}</h2>
            <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full" onClick={onClose}>
              <X className="h-4 w-4" />
              <span className="sr-only">Close parameter panel</span>
            </Button>
          </div>

          {/* Scrollable Content */}
          <div className="flex-1 overflow-y-auto p-4">{renderContent()}</div>
        </div>
      </div>
    </>
  );
}

NodeConfigPanel.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  selectedNode: PropTypes.object,
  onNodeUpdate: PropTypes.func.isRequired,
};

export default NodeConfigPanel;
