import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";

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
<<<<<<< HEAD

        {/* Input */}
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-input bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "custominput")}
          draggable
        >
          Input
        </div>

        {/* Dense */}
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-dense bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customdense")}
          draggable
        >
          Dense
        </div>

        {/* Flatten */}
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-flatten bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customflatten")}
          draggable
        >
          Flatten
        </div>

        {/* Conv2D */}
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-conv bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customconv")}
          draggable
        >
          Conv2D
        </div>

        {/* Dropout */}
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-dropout bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customdropout")}
          draggable
        >
          Dropout
        </div>

        {/* BatchNorm */}
=======
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

        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-dropout bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "customdropout")}
          draggable
        >
          Dropout
        </div>

>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)
        <div
          className="cursor-grab rounded-md border border-l-4 border-l-node-batchnorm bg-white px-3 py-2 text-xs font-medium"
          onDragStart={(e) => onDragStart(e, "custombatchnorm")}
          draggable
        >
          BatchNorm
        </div>
<<<<<<< HEAD

=======
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)
      </CardContent>
    </Card>
  );
}

export default Sidebar;
