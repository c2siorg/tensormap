import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LAYER_REGISTRY } from "../../registry/layers";

function Sidebar() {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <Card className="h-full w-48 shrink-0">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Layers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {LAYER_REGISTRY.map((layer) => (
          <div
            key={layer.type}
            className={`cursor-grab rounded-md border border-l-4 ${layer.accentClass} bg-white px-3 py-2 text-xs font-medium`}
            onDragStart={(e) => onDragStart(e, layer.type)}
            draggable
          >
            {layer.label}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export default Sidebar;
