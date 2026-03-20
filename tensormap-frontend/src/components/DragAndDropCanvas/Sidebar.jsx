import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";

const LAYER_GROUPS = [
  {
    label: "Core",
    layers: [
      { type: "custominput", display: "Input", color: "border-l-node-input" },
      { type: "customdense", display: "Dense", color: "border-l-node-dense" },
      { type: "customflatten", display: "Flatten", color: "border-l-node-flatten" },
      { type: "customconv", display: "Conv2D", color: "border-l-node-conv" },
    ],
  },
  {
    label: "Pooling",
    layers: [
      { type: "custommaxpool2d", display: "MaxPooling2D", color: "border-l-node-pooling" },
      { type: "customavgpool2d", display: "AvgPooling2D", color: "border-l-node-pooling" },
      { type: "customglobalmaxpool2d", display: "GlobalMaxPool2D", color: "border-l-node-pooling" },
      { type: "customglobalavgpool2d", display: "GlobalAvgPool2D", color: "border-l-node-pooling" },
    ],
  },
  {
    label: "Regularization",
    layers: [
      { type: "customdropout", display: "Dropout", color: "border-l-node-dropout" },
      { type: "custombatchnorm", display: "BatchNorm", color: "border-l-node-batchnorm" },
      { type: "customlayernorm", display: "LayerNorm", color: "border-l-node-batchnorm" },
    ],
  },
  {
    label: "Recurrent",
    layers: [
      { type: "customlstm", display: "LSTM", color: "border-l-node-recurrent" },
      { type: "customgru", display: "GRU", color: "border-l-node-recurrent" },
      { type: "customsimplernn", display: "SimpleRNN", color: "border-l-node-recurrent" },
      { type: "customembedding", display: "Embedding", color: "border-l-node-recurrent" },
    ],
  },
  {
    label: "Utility",
    layers: [{ type: "customreshape", display: "Reshape", color: "border-l-node-flatten" }],
  },
];

function Sidebar() {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <Card className="h-full w-48 shrink-0 overflow-y-auto">
      <CardHeader className="pb-2 pt-3">
        <CardTitle className="text-sm">Layers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 pb-4">
        {LAYER_GROUPS.map((group) => (
          <div key={group.label}>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {group.label}
            </p>
            <div className="space-y-1">
              {group.layers.map(({ type, display, color }) => (
                <div
                  key={type}
                  className={`cursor-grab rounded-md border border-l-4 ${color} bg-white px-3 py-1.5 text-xs font-medium`}
                  onDragStart={(e) => onDragStart(e, type)}
                  draggable
                >
                  {display}
                </div>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export default Sidebar;
