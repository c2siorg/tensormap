"""Static analysis of a TensorMap graph: shapes, parameter counts and diagnostics.

``analyze_graph`` takes a validated ``IRGraph``. ``analyze_canvas`` takes the raw ReactFlow
payload instead and tolerates nodes whose parameters are still invalid, which is what an editor
sends while the user is mid-edit: bad nodes become diagnostics and the rest of the graph is
still analysed.

The walk is topological. A node whose inputs are unknown is skipped without a report of its own,
so one mistake yields one diagnostic on the node that caused it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from app.ir.analysis.models import (
    Diagnostic,
    DiagnosticCode,
    GraphAnalysis,
    NodeAnalysis,
    Severity,
    Shape,
)
from app.ir.analysis.shapes import ShapeError, concat_shapes, get_rule
from app.ir.schema import IRGraph, NodeParams
from app.ir.translator import TranslationError, resolve_layer_type, translate_params_to_ir
from app.layers.registry import LAYER_REGISTRY

_FLOAT32_BYTES = 4


@dataclass(frozen=True)
class _Entry:
    """One node as the analyzer sees it. ``params`` is ``None`` when the node is invalid."""

    id: str
    layer_type: str
    name: str
    params: NodeParams | None


def analyze_graph(graph: IRGraph) -> GraphAnalysis:
    """Analyse a validated IR graph."""
    entries = [_Entry(n.id, n.node_params.layer_type, n.name, n.node_params) for n in graph.nodes]
    edges = [(e.source_id, e.target_id, e.id) for e in graph.edges]
    return _analyze(entries, edges, [])


def analyze_canvas(canvas: dict) -> GraphAnalysis:
    """Analyse a ReactFlow canvas payload (``{"nodes": [...], "edges": [...]}``).

    Nodes whose type or parameters do not validate are reported as ``invalid_params``
    diagnostics instead of aborting the analysis.
    """
    entries: list[_Entry] = []
    preset: list[Diagnostic] = []
    for n in canvas.get("nodes", []):
        node_id = str(n.get("id"))
        name = n.get("data", {}).get("name", "")
        layer_type = "unknown"
        try:
            layer_type = resolve_layer_type(n)
            params = translate_params_to_ir(layer_type, n.get("data", {}).get("params", {}))
        except TranslationError as e:
            preset.append(
                Diagnostic(
                    code=DiagnosticCode.INVALID_PARAMS,
                    severity=Severity.ERROR,
                    message=str(e),
                    node_id=node_id,
                    suggestion="Fix the highlighted layer settings.",
                )
            )
            entries.append(_Entry(node_id, layer_type, name, None))
        else:
            entries.append(_Entry(node_id, layer_type, name, params))

    edges = [
        (e.get("source"), e.get("target"), e.get("id") or f"{e.get('source')}-{e.get('target')}")
        for e in canvas.get("edges", [])
    ]
    return _analyze(entries, edges, preset)


def _label(entry: _Entry) -> str:
    spec = LAYER_REGISTRY.get(entry.layer_type)
    display = spec.display_name if spec else entry.layer_type
    name = entry.name if entry.name and entry.name != entry.layer_type else entry.id
    return f"{display} '{name}'"


def _analyze(
    entries: list[_Entry],
    raw_edges: Iterable[tuple[str | None, str | None, str]],
    diagnostics: list[Diagnostic],
) -> GraphAnalysis:
    diagnostics = list(diagnostics)
    if not entries:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.EMPTY_GRAPH,
                severity=Severity.ERROR,
                message="The canvas has no layers.",
                suggestion="Drag an Input layer onto the canvas to start.",
            )
        )
        return _finish([], diagnostics, [])

    by_id = {e.id: e for e in entries}

    # Keep only edges between known nodes; report the rest.
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target, edge_id in raw_edges:
        missing = [x for x in (source, target) if x not in by_id]
        if missing:
            anchor = next((x for x in (source, target) if x in by_id), None)
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.DANGLING_EDGE,
                    severity=Severity.ERROR,
                    message=f"Connection {edge_id} points at a layer that does not exist: {missing[0]}.",
                    node_id=anchor,
                    suggestion="Delete the connection or reconnect it to an existing layer.",
                )
            )
            continue
        incoming[target].append(source)  # type: ignore[index]
        outgoing[source].append(target)  # type: ignore[index]

    if not any(e.layer_type == "input" for e in entries):
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.NO_INPUT_NODE,
                severity=Severity.ERROR,
                message="The graph has no Input layer.",
                suggestion="Add an Input layer and connect it to the rest of the model.",
            )
        )

    order, cyclic = _topological_order(entries, incoming, outgoing)
    if cyclic:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.CYCLE,
                severity=Severity.ERROR,
                message="The graph contains a cycle; layers must form a directed acyclic graph.",
                node_id=cyclic[0],
                related_node_ids=cyclic,
                suggestion="Remove a connection that leads back to an earlier layer.",
            )
        )

    results: dict[str, NodeAnalysis] = {}
    for node_id in order:
        entry = by_id[node_id]
        results[node_id] = _analyze_node(entry, by_id, incoming.get(node_id, []), results, diagnostics)

    # Nodes caught in a cycle never became ready, so they have no shape.
    for entry in entries:
        results.setdefault(entry.id, NodeAnalysis(node_id=entry.id, layer_type=entry.layer_type))

    # A model's outputs are its leaf nodes; inputs never count, as in the Keras generator.
    outputs = [e.id for e in entries if e.layer_type != "input" and not outgoing.get(e.id)]
    return _finish([results[e.id] for e in _in_order(entries, order)], diagnostics, outputs)


def _in_order(entries: list[_Entry], order: list[str]) -> list[_Entry]:
    """Topologically ordered entries first, then anything the sort could not place."""
    by_id = {e.id: e for e in entries}
    placed = set(order)
    return [by_id[i] for i in order] + [e for e in entries if e.id not in placed]


def _topological_order(
    entries: list[_Entry], incoming: dict[str, list[str]], outgoing: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    """Kahn's algorithm, stable with respect to the canvas order.

    Returns the processable order and the ids of the nodes that sit on a cycle.
    """
    indegree = {e.id: len(incoming.get(e.id, [])) for e in entries}
    queue = deque(e.id for e in entries if indegree[e.id] == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for target in outgoing.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    placed = set(order)
    remaining = [e.id for e in entries if e.id not in placed]
    # Peel off nodes that only sit downstream of the cycle, leaving the cycle itself.
    remaining_set = set(remaining)
    changed = True
    while changed:
        changed = False
        for node_id in list(remaining_set):
            if not any(t in remaining_set for t in outgoing.get(node_id, [])):
                remaining_set.discard(node_id)
                changed = True
    return order, [i for i in remaining if i in remaining_set]


def _analyze_node(
    entry: _Entry,
    by_id: dict[str, _Entry],
    sources: list[str],
    results: dict[str, NodeAnalysis],
    diagnostics: list[Diagnostic],
) -> NodeAnalysis:
    unknown = NodeAnalysis(node_id=entry.id, layer_type=entry.layer_type)
    if entry.params is None:
        return unknown  # Already reported while translating the canvas.

    label = _label(entry)
    is_input = entry.layer_type == "input"

    if is_input:
        if sources:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.INPUT_HAS_INCOMING_EDGE,
                    severity=Severity.WARNING,
                    message=f"{label} has incoming connections, which are ignored.",
                    node_id=entry.id,
                    related_node_ids=list(sources),
                    suggestion="Delete the connections that lead into the Input layer.",
                )
            )
        input_shapes: list[Shape] = []
    else:
        if not sources:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.DISCONNECTED_NODE,
                    severity=Severity.ERROR,
                    message=f"{label} has no incoming connection.",
                    node_id=entry.id,
                    suggestion="Connect the layer to the output of an earlier layer.",
                )
            )
            return unknown
        source_results = [results.get(s) for s in sources]
        if any(r is None or r.output_shape is None for r in source_results):
            return unknown  # An upstream problem is already reported.
        input_shapes = [r.output_shape for r in source_results]  # type: ignore[union-attr, misc]

    rule = get_rule(entry.layer_type)
    if rule is None:  # A registry layer without a rule; the parity tests prevent this.
        return NodeAnalysis(node_id=entry.id, layer_type=entry.layer_type, input_shapes=input_shapes)

    try:
        if entry.layer_type == "concatenate":
            if len(input_shapes) < 2:
                raise ShapeError(
                    DiagnosticCode.MERGE_ARITY,
                    f"needs at least 2 incoming connections but has {len(input_shapes)}.",
                    "Connect another layer to the Concatenate layer, or remove it.",
                )
            rule_inputs = input_shapes
        elif is_input:
            rule_inputs = []
        elif len(input_shapes) > 1:
            # Layers other than Concatenate merge several inputs along the last axis, as Keras does.
            rule_inputs = [concat_shapes(input_shapes, -1)]
        else:
            rule_inputs = input_shapes
        result = rule(entry.params, rule_inputs)
    except ShapeError as e:
        diagnostics.append(
            Diagnostic(
                code=e.code,
                severity=e.severity,
                message=f"{label}: {e.detail}" if e.code != DiagnosticCode.MERGE_ARITY else f"{label} {e.detail}",
                node_id=entry.id,
                related_node_ids=list(sources),
                suggestion=e.suggestion,
            )
        )
        return NodeAnalysis(node_id=entry.id, layer_type=entry.layer_type, input_shapes=input_shapes)

    if result.warning is not None:
        w = result.warning
        diagnostics.append(
            Diagnostic(
                code=w.code,
                severity=w.severity,
                message=f"{label}: {w.detail}",
                node_id=entry.id,
                related_node_ids=list(sources),
                suggestion=w.suggestion,
            )
        )

    return NodeAnalysis(
        node_id=entry.id,
        layer_type=entry.layer_type,
        input_shapes=input_shapes,
        output_shape=result.output_shape,
        params=result.params,
        trainable_params=result.params - result.non_trainable,
        non_trainable_params=result.non_trainable,
        macs=result.macs,
    )


def _finish(nodes: list[NodeAnalysis], diagnostics: list[Diagnostic], outputs: list[str]) -> GraphAnalysis:
    total = sum(n.params for n in nodes)
    return GraphAnalysis(
        ok=not any(d.severity == Severity.ERROR for d in diagnostics),
        complete=bool(nodes) and all(n.output_shape is not None for n in nodes),
        nodes=nodes,
        diagnostics=diagnostics,
        output_node_ids=outputs,
        total_params=total,
        trainable_params=sum(n.trainable_params for n in nodes),
        non_trainable_params=sum(n.non_trainable_params for n in nodes),
        macs=sum(n.macs for n in nodes),
        size_bytes=total * _FLOAT32_BYTES,
    )
