/**
 * Determines whether the model can be saved.
 * Returns true when model name is filled, nodes present, all params valid, and graph connected.
 */
export const canSaveModel = (modelName, modelData) => {
  if (!modelName || modelName.trim() === "") return false;
  if (!modelData.nodes || modelData.nodes.length === 0) return false;

  for (const node of modelData.nodes) {
    if (node.type === "customdense") {
      if (
        !node.data.params.units ||
        node.data.params.units === "" ||
        !node.data.params.activation ||
        node.data.params.activation === ""
      ) {
        return false;
      }
    } else if (node.type === "custominput") {
      if (!node.data.params["dim-1"] || node.data.params["dim-1"] === "") {
        return false;
      }
    } else if (node.type === "customconv") {
      const p = node.data.params;
      if (!p.filter || !p.kernelX || !p.kernelY || !p.strideX || !p.strideY) {
        return false;
      }
    }
    // customflatten has no params to validate
  }

  return isGraphConnected(modelData);
};

const isGraphConnected = (graph) => {
  if (!graph.nodes || graph.nodes.length === 0) return false;

  const visited = new Set();
  const firstNodeId = graph.nodes[0].id;
  const queue = [firstNodeId];
  visited.add(firstNodeId);

  while (queue.length > 0) {
    const currentNodeId = queue.shift();
    const adjacentNodes = graph.edges
      .filter((edge) => edge.source === currentNodeId || edge.target === currentNodeId)
      .map((edge) => (edge.source === currentNodeId ? edge.target : edge.source));

    for (const nodeId of adjacentNodes) {
      if (!visited.has(nodeId)) {
        queue.push(nodeId);
        visited.add(nodeId);
      }
    }
  }
  return visited.size === graph.nodes.length;
};

/**
 * Strips visual-only properties from a ReactFlow graph snapshot using
 * immutable operations (no destructive delete mutations).
 */
export const generateModelJSON = (modelData) => {
  const nodes = modelData.nodes.map((node) => ({
    id: node.id,
    type: node.type,
    position: node.position,
    data: { params: node.data.params },
  }));

  const edges = modelData.edges.map((edge) => ({
    source: edge.source,
    target: edge.target,
  }));

  return { nodes, edges };
};
