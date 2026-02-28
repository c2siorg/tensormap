import PropTypes from "prop-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function Sidebar({ registry }) {
  const onDragStart = (event, layerKey) => {
    // Passing the JSON key (e.g., 'dense', 'conv2d') instead of a hardcoded type
    event.dataTransfer.setData("application/reactflow", layerKey);
    event.dataTransfer.effectAllowed = "move";
  };

  const getBorderColor = (category) => {
    switch (category) {
      case "core": return "border-l-blue-600";
      case "convolutional": return "border-l-green-600";
      default: return "border-l-slate-600";
    }
  };

  return (
    <Card className="h-full w-48 shrink-0">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Layers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {Object.entries(registry || {}).map(([layerKey, config]) => (
          <div
            key={layerKey}
            className={`cursor-grab rounded-md border border-l-4 ${getBorderColor(config.category)} bg-white px-3 py-2 text-xs font-medium`}
            onDragStart={(e) => onDragStart(e, layerKey)}
            draggable
          >
            {config.display_name}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

Sidebar.propTypes = {
  registry: PropTypes.object.isRequired,
};

export default Sidebar;