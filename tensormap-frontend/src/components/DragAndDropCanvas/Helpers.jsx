/**
 * Per-node-type validation rules.
 * Each function receives the node's params and returns true when valid.
 */
const VALIDATION_RULES = {
  customdense: (params) => !!params.units && !!params.activation,
  custominput: (params) => !!params["dim-1"],
  customconv: (params) =>
    !!(params.filter && params.kernelX && params.kernelY && params.strideX && params.strideY),
  customlstm: (params) => {
    const units = Number(params.units);
    const returnSeq = params.returnSequences;
    return units > 0 && !isNaN(units) && (returnSeq === "true" || returnSeq === "false");
  },
  customflatten: () => true,
  customdropout: () => true,
};

/**
 * Determines whether the model can be saved.
 * Returns true when model name is filled, nodes present, all params valid, and graph connected.
 */
export const canSaveModel = (modelName, modelData) => {
  if (!modelName || modelName.trim() === "") return false;
  if (!modelData.nodes || modelData.nodes.length === 0) return false;

  for (const node of modelData.nodes) {
    const validator = VALIDATION_RULES[node.type];
    if (validator && !validator(node.data.params)) return false;
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
