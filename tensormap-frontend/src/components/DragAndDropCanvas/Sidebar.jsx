import PropTypes from "prop-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function Sidebar({ registry }) {
  const onDragStart = (event, layerKey) => {
    // Passing the JSON key (e.g., 'dense', 'conv2d') instead of a hardcoded type
    event.dataTransfer.setData("application/reactflow", layerKey);
    event.dataTransfer.effectAllowed = "move";
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case "core":
        return "bg-blue-600 border-blue-700";
      case "convolutional":
        return "bg-green-600 border-green-700";
      default:
        return "bg-slate-600 border-slate-700";
    }
  };

  return (
    <Card className="h-full w-48 shrink-0">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Layers</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {Object.keys(registry || {}).length === 0 ? (
          <p className="text-xs text-muted-foreground">Loading layers...</p>
        ) : (
          Object.entries(registry).map(([layerKey, config]) => (
            <div
              key={layerKey}
              className={`cursor-grab rounded border px-3 py-2 text-sm text-white shadow-sm active:cursor-grabbing ${getCategoryColor(config.category)}`}
              onDragStart={(event) => onDragStart(event, layerKey, config)}
              draggable
            >
              {config.display_name}
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}

Sidebar.propTypes = {
  registry: PropTypes.objectOf(
    PropTypes.shape({
      display_name: PropTypes.string.isRequired,
      category: PropTypes.string.isRequired,
      params: PropTypes.object,
    }),
  ).isRequired,
};

export default Sidebar;
