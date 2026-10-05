"""Static graph analysis: shape inference, parameter counts and diagnostics, without TensorFlow."""

from app.ir.analysis.analyzer import analyze_canvas, analyze_graph
from app.ir.analysis.models import (
    Diagnostic,
    DiagnosticCode,
    GraphAnalysis,
    NodeAnalysis,
    Severity,
    Shape,
)

__all__ = [
    "Diagnostic",
    "DiagnosticCode",
    "GraphAnalysis",
    "NodeAnalysis",
    "Severity",
    "Shape",
    "analyze_canvas",
    "analyze_graph",
]
