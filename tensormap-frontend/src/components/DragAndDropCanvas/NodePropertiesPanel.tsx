/**
 * NodePropertiesPanel - Dynamic property editor for selected nodes.
 *
 * Phase 1 Week 2 - Fully registry-driven implementation.
 * Generates form fields dynamically based on ParamSpec from the registry.
 */

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { HelpCircle, RotateCcw } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getLayerSpec } from "../../hooks/useLayerRegistry";
import type { ParamSpec } from "../../types/registry";
import { LEGACY_TYPE_MAP } from "../../types/registry";

interface NodePropertiesPanelProps {
  selectedNode: any | null;
  modelName: string;
  onModelNameChange: (name: string) => void;
  onSave: () => void;
  canSave: boolean;
  onNodeUpdate: (nodeId: string, newParams: Record<string, any>) => void;
}

/**
 * Render an input field for an integer parameter.
 */
interface ParamInputProps {
  paramName: string;
  spec: ParamSpec;
  value: any;
  onChange: (value: any) => void;
  error?: string;
}

const IntInput = ({ paramName, spec, value, onChange, error }: ParamInputProps) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (val === "" || val === null) {
      onChange(null);
    } else {
      const num = parseInt(val, 10);
      if (!isNaN(num)) {
        onChange(num);
      }
    }
  };

  return (
    <div className="space-y-1">
      <Label className="text-xs flex items-center gap-1">
        {paramName}
        {spec.required && <span className="text-red-500">*</span>}
        {spec.description && (
          <HelpCircle className="h-3 w-3 text-gray-400" title={spec.description} />
        )}
      </Label>
      <Input
        type="number"
        step="1"
        min={spec.min ?? undefined}
        max={spec.max ?? undefined}
        placeholder={spec.default !== null ? String(spec.default) : ""}
        value={value ?? ""}
        onChange={handleChange}
        className={error ? "border-red-500" : ""}
      />
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );
};

/**
 * Render an input field for a float parameter.
 */
const FloatInput = ({ paramName, spec, value, onChange, error }: ParamInputProps) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (val === "" || val === null) {
      onChange(null);
    } else {
      const num = parseFloat(val);
      if (!isNaN(num)) {
        onChange(num);
      }
    }
  };

  return (
    <div className="space-y-1">
      <Label className="text-xs flex items-center gap-1">
        {paramName}
        {spec.required && <span className="text-red-500">*</span>}
        {spec.description && (
          <HelpCircle className="h-3 w-3 text-gray-400" title={spec.description} />
        )}
      </Label>
      <Input
        type="number"
        step="0.01"
        min={spec.min ?? undefined}
        max={spec.max ?? undefined}
        placeholder={spec.default !== null ? String(spec.default) : ""}
        value={value ?? ""}
        onChange={handleChange}
        className={error ? "border-red-500" : ""}
      />
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );
};

/**
 * Render a checkbox for a boolean parameter.
 */
const BoolInput = ({ paramName, spec, value, onChange }: ParamInputProps) => {
  return (
    <div className="space-y-1">
      <Label className="text-xs flex items-center gap-2">
        <input
          type="checkbox"
          checked={value ?? spec.default ?? false}
          onChange={(e) => onChange(e.target.checked)}
          className="rounded border-gray-300"
        />
        {paramName}
        {spec.description && (
          <HelpCircle className="h-3 w-3 text-gray-400" title={spec.description} />
        )}
      </Label>
    </div>
  );
};

/**
 * Render a select dropdown for an enum parameter.
 */
const EnumInput = ({ paramName, spec, value, onChange, error }: ParamInputProps) => {
  const options = spec.values || [];

  return (
    <div className="space-y-1">
      <Label className="text-xs flex items-center gap-1">
        {paramName}
        {spec.required && <span className="text-red-500">*</span>}
        {spec.description && (
          <HelpCircle className="h-3 w-3 text-gray-400" title={spec.description} />
        )}
      </Label>
      <Select value={value ?? spec.default ?? ""} onValueChange={(v) => onChange(v || null)}>
        <SelectTrigger className={error ? "border-red-500" : ""}>
          <SelectValue placeholder="Select..." />
        </SelectTrigger>
        <SelectContent>
          {options.map((opt) => (
            <SelectItem key={opt} value={opt}>
              {opt}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );
};

/**
 * Validate a parameter value against its spec.
 */
function validateParam(paramName: string, value: any, spec: ParamSpec): string | null {
  // Check required
  if (spec.required && (value === null || value === undefined || value === "")) {
    return `${paramName} is required`;
  }

  // Skip validation if empty and not required
  if (value === null || value === undefined || value === "") {
    return null;
  }

  // Type-specific validation
  if (spec.type === "int") {
    const num = Number(value);
    if (!Number.isInteger(num)) {
      return `${paramName} must be an integer`;
    }
    if (spec.min !== undefined && spec.min !== null && num < spec.min) {
      return `${paramName} must be >= ${spec.min}`;
    }
    if (spec.max !== undefined && spec.max !== null && num > spec.max) {
      return `${paramName} must be <= ${spec.max}`;
    }
  }

  if (spec.type === "float") {
    const num = Number(value);
    if (isNaN(num)) {
      return `${paramName} must be a number`;
    }
    if (spec.min !== undefined && spec.min !== null && num < spec.min) {
      return `${paramName} must be >= ${spec.min}`;
    }
    if (spec.max !== undefined && spec.max !== null && num > spec.max) {
      return `${paramName} must be <= ${spec.max}`;
    }
  }

  return null;
}

function NodePropertiesPanel({
  selectedNode,
  modelName,
  onModelNameChange,
  onSave,
  canSave,
  onNodeUpdate,
}: NodePropertiesPanelProps) {
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  // If no node selected, show the model save panel
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

  const { data, id } = selectedNode;
  const params = data.params || {};

  // Determine the layer type
  // Old nodes use 'type' directly (custominput, customdense, etc.)
  // New nodes use data.layerType (input, dense, etc.)
  const layerType = data.layerType || selectedNode.type;

  const registryType = LEGACY_TYPE_MAP[layerType] || layerType;
  const layerSpec = getLayerSpec(registryType);

  // If layer spec not found, show error
  if (!layerSpec) {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm text-red-500">Unknown Layer</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-gray-600">
            Layer type &quot;{layerType}&quot; not found in registry.
          </p>
        </CardContent>
      </Card>
    );
  }

  const handleParamChange = (paramName: string, value: any) => {
    const newParams = { ...params, [paramName]: value };
    onNodeUpdate(id, newParams);

    // Validate the new value
    const spec = layerSpec.params[paramName];
    if (spec) {
      const error = validateParam(paramName, value, spec);
      setValidationErrors((prev) => {
        if (error) {
          return { ...prev, [paramName]: error };
        } else {
          const { [paramName]: _, ...rest } = prev;
          return rest;
        }
      });
    }
  };

  const handleResetToDefaults = () => {
    const defaults: Record<string, any> = {};
    Object.entries(layerSpec.params).forEach(([paramName, spec]) => {
      defaults[paramName] = spec.default ?? null;
    });
    onNodeUpdate(id, defaults);
    setValidationErrors({});
  };

  // Render parameter inputs
  const paramInputs = Object.entries(layerSpec.params).map(([paramName, spec]) => {
    const value = params[paramName];
    const error = validationErrors[paramName];

    const commonProps = {
      paramName,
      spec,
      value,
      onChange: (val: any) => handleParamChange(paramName, val),
      error,
    };

    switch (spec.type) {
      case "int":
        return <IntInput key={paramName} {...commonProps} />;
      case "float":
        return <FloatInput key={paramName} {...commonProps} />;
      case "bool":
        return <BoolInput key={paramName} {...commonProps} />;
      case "enum":
        return <EnumInput key={paramName} {...commonProps} />;
      default:
        return null;
    }
  });

  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="text-sm">{layerSpec.display_name} Layer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {paramInputs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No configurable parameters</p>
        ) : (
          <>
            {paramInputs}
            <Button
              variant="outline"
              size="sm"
              className="w-full mt-2"
              onClick={handleResetToDefaults}
            >
              <RotateCcw className="h-3 w-3 mr-1" />
              Reset to Defaults
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default NodePropertiesPanel;
