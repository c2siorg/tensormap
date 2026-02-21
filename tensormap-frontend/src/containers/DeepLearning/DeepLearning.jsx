import Canvas from "../../components/DragAndDropCanvas/Canvas";

/**
 * Model builder page that wraps the drag-and-drop ReactFlow canvas.
 *
 * Provides the visual neural network design interface where users
 * assemble layers, configure hyperparameters, and validate models.
 */
function DeepLearning() {
  return (
    <div className="space-y-4">
      <Canvas />
    </div>
  );
}

export default DeepLearning;
