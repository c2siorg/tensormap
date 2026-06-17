"""
Intermediate Representation (IR) Schema for TensorMap Neural Network Graphs.

This module defines the Pydantic models for representing neural network architectures
as directed acyclic graphs (DAGs). The IR serves as the contract between the frontend
ReactFlow canvas and the backend Keras model generation.

Schema Diagram:
┌─────────────────────────────────────────────────────────────┐
│ IRGraph                                                     │
│ ├── version: str                                            │
│ ├── nodes: list[IRNode]                                     │
│ └── edges: list[IREdge]                                     │
└─────────────────────────────────────────────────────────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────────┐          ┌──────────────────┐
│ IRNode              │          │ IREdge           │
│ ├── id: str         │          │ ├── id: str      │
│ ├── node_params ────┼──┐       │ ├── source_id    │
│ ├── position        │  │       │ ├── target_id    │
│ └── input_node_ids  │  │       │ ├── source_handle│
└─────────────────────┘  │       │ └── target_handle│
                         │       └──────────────────┘
                         ▼
        ┌────────────────────────────────────────┐
        │ NodeParams (Discriminated Union)       │
        │ ├── InputParams (layer_type="input")   │
        │ ├── DenseParams (layer_type="dense")   │
        │ ├── Conv2DParams (layer_type="conv2d") │
        │ ├── ... (15 layer types total)         │
        │ └── ConcatenateParams                  │
        └────────────────────────────────────────┘

The discriminated union pattern uses the `layer_type` field to automatically
select the correct parameter model during deserialization.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Per-Layer Parameter Models
# ============================================================================
# Each layer type has its own Pydantic model with a literal layer_type field
# for discriminated union matching.

DenseActivation = Literal[
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
    "linear",
    "elu",
    "selu",
    "softplus",
    "softsign",
    "swish",
    "gelu",
    "exponential",
]


class InputParams(BaseModel):
    """Parameters for Input layer."""

    layer_type: Literal["input"]
    shape: int = Field(..., gt=0, description="Input dimensions (e.g., 784 for 28x28 flattened)")


class DenseParams(BaseModel):
    """Parameters for Dense (fully connected) layer."""

    layer_type: Literal["dense"]
    units: int = Field(..., gt=0, description="Number of neurons")
    activation: DenseActivation = Field(default="relu", description="Activation function")


class FlattenParams(BaseModel):
    """Parameters for Flatten layer (no configurable params)."""

    layer_type: Literal["flatten"]


class Conv2DParams(BaseModel):
    """Parameters for Conv2D layer."""

    layer_type: Literal["conv2d"]
    filters: int = Field(..., gt=0, description="Number of convolutional filters")
    kernel_size: int = Field(default=3, gt=0, description="Kernel size (e.g., 3 for 3x3)")
    strides: int = Field(default=1, gt=0, description="Stride length")
    padding: Literal["valid", "same"] = Field(default="same", description="Padding mode")
    activation: DenseActivation = Field(default="relu", description="Activation function")


class MaxPool2DParams(BaseModel):
    """Parameters for MaxPooling2D layer."""

    layer_type: Literal["maxpool2d"]
    pool_size: int = Field(default=2, gt=0, description="Pooling window size")
    strides: int = Field(default=2, gt=0, description="Stride length")
    padding: Literal["valid", "same"] = Field(default="valid", description="Padding mode")


class AvgPool2DParams(BaseModel):
    """Parameters for AveragePooling2D layer."""

    layer_type: Literal["avgpool2d"]
    pool_size: int = Field(default=2, gt=0, description="Pooling window size")
    strides: int = Field(default=2, gt=0, description="Stride length")
    padding: Literal["valid", "same"] = Field(default="valid", description="Padding mode")


class GlobalAvgPool2DParams(BaseModel):
    """Parameters for GlobalAveragePooling2D layer (no configurable params)."""

    layer_type: Literal["globalavgpool2d"]


class LSTMParams(BaseModel):
    """Parameters for LSTM layer."""

    layer_type: Literal["lstm"]
    units: int = Field(..., gt=0, description="Number of LSTM units")
    return_sequences: bool = Field(default=False, description="Return full sequence or last output only")
    activation: Literal["tanh", "sigmoid", "relu"] = Field(default="tanh", description="Activation function")


class GRUParams(BaseModel):
    """Parameters for GRU layer."""

    layer_type: Literal["gru"]
    units: int = Field(..., gt=0, description="Number of GRU units")
    return_sequences: bool = Field(default=False, description="Return full sequence or last output only")
    activation: Literal["tanh", "sigmoid", "relu"] = Field(default="tanh", description="Activation function")


class SimpleRNNParams(BaseModel):
    """Parameters for SimpleRNN layer."""

    layer_type: Literal["simplernn"]
    units: int = Field(..., gt=0, description="Number of RNN units")
    return_sequences: bool = Field(default=False, description="Return full sequence or last output only")
    activation: Literal["tanh", "sigmoid", "relu"] = Field(default="tanh", description="Activation function")


class EmbeddingParams(BaseModel):
    """Parameters for Embedding layer."""

    layer_type: Literal["embedding"]
    input_dim: int = Field(..., gt=0, description="Vocabulary size")
    output_dim: int = Field(..., gt=0, description="Embedding dimension")


class DropoutParams(BaseModel):
    """Parameters for Dropout layer."""

    layer_type: Literal["dropout"]
    rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout rate (0.0 to 1.0)")


class BatchNormParams(BaseModel):
    """Parameters for BatchNormalization layer."""

    layer_type: Literal["batchnorm"]
    momentum: float = Field(default=0.99, ge=0.0, le=1.0, description="Momentum for moving average")
    epsilon: float = Field(default=0.001, ge=1e-7, description="Small constant for numerical stability")


class ReshapeParams(BaseModel):
    """Parameters for Reshape layer."""

    layer_type: Literal["reshape"]
    target_shape: str = Field(..., description="Target shape as comma-separated integers (e.g., '7,7,64')")


class ConcatenateParams(BaseModel):
    """Parameters for Concatenate layer (merge layer)."""

    layer_type: Literal["concatenate"]
    axis: int = Field(default=-1, description="Axis along which to concatenate")


# ============================================================================
# Discriminated Union for All Layer Types
# ============================================================================

NodeParams = Annotated[
    InputParams
    | DenseParams
    | FlattenParams
    | Conv2DParams
    | MaxPool2DParams
    | AvgPool2DParams
    | GlobalAvgPool2DParams
    | LSTMParams
    | GRUParams
    | SimpleRNNParams
    | EmbeddingParams
    | DropoutParams
    | BatchNormParams
    | ReshapeParams
    | ConcatenateParams,
    Field(discriminator="layer_type"),
]


# ============================================================================
# Graph Structure Models
# ============================================================================


class IRNode(BaseModel):
    """
    A single node in the neural network graph.

    Attributes:
        id: Unique identifier for this node (e.g., "node-1", "node-2")
        node_params: Layer-specific parameters (discriminated by layer_type)
        name: Human-readable name for the node (e.g., "Input Layer", "Dense 1")
        position: Canvas position for ReactFlow (x, y coordinates)
        input_node_ids: List of parent node IDs (empty for single-parent, 2+ for merge layers)
    """

    id: str
    node_params: NodeParams
    name: str = ""
    position: dict[str, float] = Field(default_factory=dict, description="Canvas position {x, y}")
    input_node_ids: list[str] = Field(default_factory=list, description="Parent node IDs for this layer")


class IREdge(BaseModel):
    """
    A directed edge connecting two nodes in the graph.

    Attributes:
        id: Unique identifier for this edge (e.g., "edge-1-2")
        source_id: ID of the source node
        target_id: ID of the target node
        source_handle: Optional handle ID on source node (for multi-output nodes)
        target_handle: Optional handle ID on target node (for multi-input nodes like Concatenate)
    """

    id: str
    source_id: str
    target_id: str
    source_handle: str | None = None
    target_handle: str | None = None


class IRGraph(BaseModel):
    """
    Complete neural network graph representation.

    Attributes:
        version: Schema version for forward compatibility
        nodes: List of all nodes in the graph
        edges: List of all edges connecting nodes
    """

    version: str = "1.0"
    nodes: list[IRNode] = Field(..., min_length=1, description="At least one node required")
    edges: list[IREdge] = Field(default_factory=list, description="Edges connecting nodes")

    def to_reactflow_json(self) -> dict:
        nodes = []
        edges = []
        for n in self.nodes:
            params = n.node_params.model_dump()
            layer_type = params.pop("layer_type")
            nodes.append(
                {
                    "id": n.id,
                    "type": layer_type + "Node",
                    "position": n.position,
                    "data": {"name": n.name or layer_type.title(), "params": params},
                }
            )
        for e in self.edges:
            edges.append(
                {
                    "id": e.id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "sourceHandle": e.source_handle,
                    "targetHandle": e.target_handle,
                }
            )
        return {"nodes": nodes, "edges": edges}


# ============================================================================
# Validation
# ============================================================================


class IRValidationError(ValueError):
    """Custom exception for invalid IR graphs."""

    def __init__(self, message: str, node_id: str | None = None, field_name: str | None = None):
        super().__init__(message)
        self.message = message
        self.node_id = node_id
        self.field_name = field_name

    def to_dict(self):
        return {"message": self.message, "node_id": self.node_id, "field_name": self.field_name}


def validate_ir_graph(graph: IRGraph) -> list[IRValidationError]:
    """
    Validate the IR graph structure and constraints.

    Checks:
    1. Exactly one input node exists (layer_type == "input")
    2. Concatenate nodes have at least 2 incoming edges
    3. No cycles exist in the graph (DAG constraint)
    4. All node IDs referenced in edges exist in nodes list
    5. Reshape nodes have valid target_shape

    Args:
        graph: The IR graph to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check 1: Exactly one input node must exist
    inputs = [n for n in graph.nodes if n.node_params.layer_type == "input"]
    if len(inputs) == 0:
        errors.append(IRValidationError(message="Graph must contain exactly one input node (layer_type='input')"))
    elif len(inputs) > 1:
        errors.append(IRValidationError(message="Graph has multiple input nodes. Only one is allowed."))

    # Build node ID set for reference checking
    node_ids = {node.id for node in graph.nodes}

    # Check 4: All edge references must point to existing nodes
    for edge in graph.edges:
        if edge.source_id not in node_ids:
            errors.append(
                IRValidationError(
                    message=f"Edge {edge.id} references non-existent source node: {edge.source_id}",
                    node_id=edge.source_id,
                )
            )
        if edge.target_id not in node_ids:
            errors.append(
                IRValidationError(
                    message=f"Edge {edge.id} references non-existent target node: {edge.target_id}",
                    node_id=edge.target_id,
                )
            )

    # Build adjacency list for graph traversal
    incoming_edges: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    outgoing_edges: dict[str, list[str]] = {node.id: [] for node in graph.nodes}

    for edge in graph.edges:
        if edge.source_id in node_ids and edge.target_id in node_ids:
            incoming_edges[edge.target_id].append(edge.source_id)
            outgoing_edges[edge.source_id].append(edge.target_id)

    # Check 2: Concatenate nodes must have at least 2 incoming edges
    for node in graph.nodes:
        if node.node_params.layer_type == "concatenate" and len(incoming_edges[node.id]) < 2:
            errors.append(
                IRValidationError(
                    message=f"Concatenate node must have at least 2 incoming edges, "
                    f"but has {len(incoming_edges[node.id])}",
                    node_id=node.id,
                )
            )

    # Check 3: No cycles (DAG constraint) using DFS
    visited = set()
    rec_stack = set()

    def has_cycle(node_id: str) -> bool:
        """DFS helper to detect cycles."""
        visited.add(node_id)
        rec_stack.add(node_id)

        for neighbor in outgoing_edges[node_id]:
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node_id)
        return False

    for node in graph.nodes:
        if node.id not in visited and has_cycle(node.id):
            errors.append(IRValidationError(message="Graph contains a cycle (must be a directed acyclic graph)"))
            break  # One cycle error is enough

    # Check 5: Reshape nodes have valid target_shape
    for node in graph.nodes:
        if node.node_params.layer_type == "reshape":
            shape = getattr(node.node_params, "target_shape", None)
            if not shape:
                errors.append(IRValidationError(message="Reshape node requires a target_shape", node_id=node.id))

    return errors
