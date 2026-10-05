"""Result types for static graph analysis.

These are plain Pydantic models so the same objects serve the Python callers and the
``POST /layers/analyze-graph`` response.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# A tensor shape without the batch axis. Every dimension is known: the IR has no dynamic axes.
Shape = tuple[int, ...]


class Severity(StrEnum):
    """How a diagnostic affects the model."""

    ERROR = "error"  # The model cannot be built.
    WARNING = "warning"  # The model builds, but the layout is probably not what the user meant.


class DiagnosticCode(StrEnum):
    """Stable machine-readable identifiers, so the UI can react without parsing messages."""

    EMPTY_GRAPH = "empty_graph"
    NO_INPUT_NODE = "no_input_node"
    INVALID_PARAMS = "invalid_params"
    DANGLING_EDGE = "dangling_edge"
    CYCLE = "cycle"
    DISCONNECTED_NODE = "disconnected_node"
    MERGE_ARITY = "merge_arity"
    RANK_MISMATCH = "rank_mismatch"
    SHAPE_MISMATCH = "shape_mismatch"
    INVALID_OUTPUT_SIZE = "invalid_output_size"
    INVALID_RESHAPE = "invalid_reshape"
    INPUT_HAS_INCOMING_EDGE = "input_has_incoming_edge"
    UNFLATTENED_DENSE = "unflattened_dense"


class Diagnostic(BaseModel):
    """One problem found in the graph, attached to the node that causes it."""

    code: DiagnosticCode
    severity: Severity
    message: str
    node_id: str | None = None
    suggestion: str | None = None
    related_node_ids: list[str] = Field(default_factory=list)


class NodeAnalysis(BaseModel):
    """What the analyzer inferred for one node.

    ``output_shape`` is ``None`` when it cannot be known, either because this node is invalid
    or because something upstream of it is. Nothing downstream of an invalid node is reported
    again, so one mistake produces one diagnostic rather than a cascade.
    """

    node_id: str
    layer_type: str
    input_shapes: list[Shape] = Field(default_factory=list)
    output_shape: Shape | None = None
    params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    macs: int = 0


class GraphAnalysis(BaseModel):
    """The result of analysing a whole graph."""

    ok: bool = Field(description="True when there are no error-level diagnostics.")
    complete: bool = Field(description="True when every node has a known output shape.")
    nodes: list[NodeAnalysis] = Field(default_factory=list, description="In topological order.")
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    macs: int = 0
    size_bytes: int = Field(default=0, description="Weight storage as float32.")

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]

    def node(self, node_id: str) -> NodeAnalysis | None:
        return next((n for n in self.nodes if n.node_id == node_id), None)
