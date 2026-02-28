/**
 * Determines whether the model can be saved dynamically based on the Layer Registry contract.
 * Returns true when model name is filled, nodes present, all required params valid, and graph connected.
 */
export const canSaveModel = (modelName, modelData) => {
  if (!modelName || modelName.trim() === "") return false;
  if (!modelData.nodes || modelData.nodes.length === 0) return false;

  for (const node of modelData.nodes) {
    const params = node.data.params || {};
    const registry = node.data.registry || {};
    const registryParams = registry.params || {};

    // Dynamically validate only the parameters the API explicitly marked as "required: true"
    for (const [paramKey, paramConfig] of Object.entries(registryParams)) {
      if (paramConfig.required) {
        const val = params[paramKey];
        if (val === undefined || val === null || val === "") {
          return false;
        }
      }
    }
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
 * Strips visual-only properties from a ReactFlow graph snapshot,
 * but specifically PRESERVES the registry contract for the backend generator.
 */
export const generateModelJSON = (modelData) => {
  const nodes = modelData.nodes.map((node) => ({
    id: node.id,
    type: node.type,
    position: node.position,
    data: { 
      params: node.data.params,
      registry: node.data.registry // Send the contract to backend
    },
  }));

  const edges = modelData.edges.map((edge) => ({
    source: edge.source,
    target: edge.target,
  }));

  return { nodes, edges };
};