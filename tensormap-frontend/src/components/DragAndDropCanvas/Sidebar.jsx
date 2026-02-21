import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-input bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "custominput")}
          draggable
        >
          Input
        </div>
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-dense bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customdense")}
          draggable
        >
          Dense
        </div>
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-flatten bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customflatten")}
          draggable
        >
          Flatten
        </div>
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-conv bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customconv")}
          draggable
        >
          Conv2D
        </div>
      </CardContent>
    </Card>
  );
}

export default Sidebar;
