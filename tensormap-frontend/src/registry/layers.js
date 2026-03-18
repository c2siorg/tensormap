export const LAYER_REGISTRY = [
  {
    type: "custominput",
    label: "Input",
    accentClass: "border-l-node-input",
    defaultParams: { "dim-1": "", "dim-2": "", "dim-3": "" },
    requiredParams: ["dim-1"],
  },
  {
    type: "customdense",
    label: "Dense",
    accentClass: "border-l-node-dense",
    defaultParams: { units: "", activation: "" },
    requiredParams: ["units", "activation"],
  },
  {
    type: "customflatten",
    label: "Flatten",
    accentClass: "border-l-node-flatten",
    defaultParams: {},
    requiredParams: [],
  },
  {
    type: "customconv",
    label: "Conv2D",
    accentClass: "border-l-node-conv",
    defaultParams: {
      filter: "",
      padding: "valid",
      activation: "none",
      strideX: "",
      strideY: "",
      kernelX: "",
      kernelY: "",
    },
    requiredParams: ["filter", "kernelX", "kernelY", "strideX", "strideY"],
  },
];

export const getLayerByType = (type) => LAYER_REGISTRY.find((layer) => layer.type === type);
