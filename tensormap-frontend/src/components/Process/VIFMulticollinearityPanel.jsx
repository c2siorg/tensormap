import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getVIFMetrics } from "../../services/FileServices";
import logger from "../../shared/logger";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Alert, AlertDescription } from "../ui/alert";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import {
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Info,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";

/**
 * Diagnostic panel computing Variance Inflation Factor (VIF) and tabular multicollinearity metrics.
 *
 * Highlights severe multicollinearity (VIF >= 10), moderate collinearity (5 <= VIF < 10),
 * and condition index of the feature matrix, providing automated remediation advice.
 *
 * @param {{ fileId: string }} props
 */
const VIFMulticollinearityPanel = ({ fileId }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filterSeverity, setFilterSeverity] = useState("ALL");
  const [showGuide, setShowGuide] = useState(false);

  const fetchVIF = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getVIFMetrics(id);
      setData(res);
    } catch (err) {
      logger.error("Error fetching VIF metrics:", err);
      setError(err?.response?.data?.message || err?.message || "Failed to load VIF metrics");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (fileId) {
      fetchVIF(fileId);
    } else {
      setData(null);
      setError(null);
    }
  }, [fileId, fetchVIF]);

  const summary = data?.summary || {};
  const vifScores = data?.vif_scores || [];
  const constantColumns = data?.constant_columns || [];

  const filteredScores = vifScores.filter((item) => {
    if (filterSeverity === "ALL") return true;
    if (filterSeverity === "HIGH") {
      return item.severity === "HIGH" || item.severity === "PERFECT_COLLINEARITY";
    }
    return item.severity === filterSeverity;
  });

  const getOverallBadge = (severity) => {
    switch (severity) {
      case "HIGH":
        return (
          <Badge variant="destructive" className="flex items-center gap-1 font-semibold">
            <ShieldAlert className="h-3.5 w-3.5" /> High Multicollinearity
          </Badge>
        );
      case "MODERATE":
        return (
          <Badge className="flex items-center gap-1 bg-amber-500 hover:bg-amber-600 text-white font-semibold">
            <AlertTriangle className="h-3.5 w-3.5" /> Moderate Multicollinearity
          </Badge>
        );
      default:
        return (
          <Badge className="flex items-center gap-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold">
            <CheckCircle2 className="h-3.5 w-3.5" /> Low Multicollinearity
          </Badge>
        );
    }
  };

  const getSeverityBadge = (severity) => {
    switch (severity) {
      case "PERFECT_COLLINEARITY":
        return (
          <Badge variant="destructive" className="bg-red-700 text-white">
            Perfect
          </Badge>
        );
      case "HIGH":
        return <Badge variant="destructive">High (VIF &ge; 10)</Badge>;
      case "MODERATE":
        return (
          <Badge className="bg-amber-500 hover:bg-amber-600 text-white">
            Moderate (5&ndash;10)
          </Badge>
        );
      default:
        return (
          <Badge className="bg-emerald-600 hover:bg-emerald-700 text-white">Low (&lt; 5)</Badge>
        );
    }
  };

  return (
    <div className="space-y-4" data-testid="vif-multicollinearity-panel">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm text-foreground">Multicollinearity Profiler</span>
          {summary.overall_multicollinearity && getOverallBadge(summary.overall_multicollinearity)}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowGuide((prev) => !prev)}
            className="h-8 gap-1 text-xs text-muted-foreground"
            data-testid="toggle-vif-guide-btn"
          >
            <HelpCircle className="h-3.5 w-3.5" />
            {showGuide ? "Hide Guide" : "VIF Guide"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchVIF(fileId)}
            disabled={loading || !fileId}
            className="h-8 gap-1 text-xs"
            data-testid="refresh-vif-btn"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {showGuide && (
        <div
          className="rounded-lg border bg-muted/40 p-4 text-xs space-y-2"
          data-testid="vif-guide-panel"
        >
          <h4 className="font-semibold text-sm flex items-center gap-1.5 text-foreground">
            <Info className="h-4 w-4 text-blue-500" />
            Understanding Variance Inflation Factor (VIF) & Multicollinearity
          </h4>
          <p className="text-muted-foreground">
            <strong>Variance Inflation Factor (VIF)</strong> quantifies how much the variance of an
            estimated regression coefficient is inflated due to collinearity with other explanatory
            features:
            <span className="ml-1 font-mono">VIF_j = 1 / (1 - R_j^2)</span>, where{" "}
            <span className="font-mono">R_j^2</span> is the coefficient of determination from
            regressing feature <span className="font-mono">j</span> on all other features.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1">
            <div className="rounded border bg-card p-2">
              <span className="font-semibold text-emerald-600 block">VIF &lt; 5 (Healthy)</span>
              <span className="text-muted-foreground">
                Features are largely independent. Estimated model weights remain stable and
                interpretable.
              </span>
            </div>
            <div className="rounded border bg-card p-2">
              <span className="font-semibold text-amber-500 block">
                5 &le; VIF &lt; 10 (Moderate)
              </span>
              <span className="text-muted-foreground">
                Noticeable correlation. Feature coefficients exhibit inflated variance; assess
                importance instability.
              </span>
            </div>
            <div className="rounded border bg-card p-2">
              <span className="font-semibold text-destructive block">VIF &ge; 10 (Severe)</span>
              <span className="text-muted-foreground">
                High multicollinearity. Dropping redundant columns, combining via PCA, or applying
                L1/L2 penalties is recommended.
              </span>
            </div>
          </div>
          <p className="text-muted-foreground pt-1">
            <strong>Condition Index:</strong> Measures feature matrix ill-conditioning via singular
            values (<span className="font-mono">&lambda;_max / &lambda;_min</span>). Values &gt; 15
            indicate moderate collinearity; values &gt; 30 signal severe multicollinearity.
          </p>
        </div>
      )}

      {loading && !data && (
        <div className="flex h-28 items-center justify-center text-sm text-muted-foreground">
          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
          Computing Variance Inflation Factors and Condition Index...
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="flex items-center justify-between">
            <span>{error}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchVIF(fileId)}
              className="ml-3 h-7 text-xs"
            >
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {!loading && !error && data && vifScores.length === 0 && (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            At least 2 non-constant numeric features are required to compute Variance Inflation
            Factors.
            {constantColumns.length > 0 && (
              <span className="block mt-1 font-mono text-xs">
                Zero-variance constant columns detected: {constantColumns.join(", ")}
              </span>
            )}
          </AlertDescription>
        </Alert>
      )}

      {!loading && !error && data && vifScores.length > 0 && (
        <>
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border bg-card p-3 shadow-sm">
              <span className="text-xs font-medium text-muted-foreground">Evaluated Features</span>
              <div className="mt-1 text-2xl font-bold">
                {summary.total_features_evaluated ?? vifScores.length}
              </div>
              <span className="text-[11px] text-muted-foreground">Non-constant numeric</span>
            </div>
            <div className="rounded-lg border bg-card p-3 shadow-sm">
              <span className="text-xs font-medium text-muted-foreground">Condition Index</span>
              <div className="mt-1 text-2xl font-bold font-mono">
                {summary.condition_index != null ? summary.condition_index : "—"}
              </div>
              <span className="text-[11px] text-muted-foreground">
                {summary.condition_index >= 30
                  ? "Severe ill-conditioning"
                  : summary.condition_index >= 15
                    ? "Moderate ill-conditioning"
                    : "Well-conditioned matrix"}
              </span>
            </div>
            <div className="rounded-lg border bg-card p-3 shadow-sm">
              <span className="text-xs font-medium text-muted-foreground">High VIF (&ge; 10)</span>
              <div
                className={`mt-1 text-2xl font-bold ${summary.high_vif_features > 0 ? "text-destructive" : "text-foreground"}`}
              >
                {summary.high_vif_features ?? 0}
              </div>
              <span className="text-[11px] text-muted-foreground">Requires remediation</span>
            </div>
            <div className="rounded-lg border bg-card p-3 shadow-sm">
              <span className="text-xs font-medium text-muted-foreground">
                Moderate VIF (5&ndash;10)
              </span>
              <div
                className={`mt-1 text-2xl font-bold ${summary.moderate_vif_features > 0 ? "text-amber-500" : "text-foreground"}`}
              >
                {summary.moderate_vif_features ?? 0}
              </div>
              <span className="text-[11px] text-muted-foreground">
                Monitor coefficient variance
              </span>
            </div>
          </div>

          {constantColumns.length > 0 && (
            <div className="rounded border border-amber-200 bg-amber-50/50 dark:bg-amber-950/20 p-2.5 text-xs text-amber-800 dark:text-amber-200">
              <span className="font-semibold">Notice:</span> Excluded constant columns with zero
              variance: <span className="font-mono">{constantColumns.join(", ")}</span>
            </div>
          )}

          {/* Filter Pills */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex items-center gap-1.5 text-xs">
              <span className="font-medium text-muted-foreground">Filter by Severity:</span>
              <Button
                variant={filterSeverity === "ALL" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setFilterSeverity("ALL")}
                className="h-7 text-xs px-2.5"
                data-testid="filter-vif-all"
              >
                All ({vifScores.length})
              </Button>
              <Button
                variant={filterSeverity === "HIGH" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setFilterSeverity("HIGH")}
                className="h-7 text-xs px-2.5 text-destructive"
                data-testid="filter-vif-high"
              >
                High ({summary.high_vif_features ?? 0})
              </Button>
              <Button
                variant={filterSeverity === "MODERATE" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setFilterSeverity("MODERATE")}
                className="h-7 text-xs px-2.5 text-amber-600"
                data-testid="filter-vif-mod"
              >
                Moderate ({summary.moderate_vif_features ?? 0})
              </Button>
              <Button
                variant={filterSeverity === "LOW" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setFilterSeverity("LOW")}
                className="h-7 text-xs px-2.5 text-emerald-600"
                data-testid="filter-vif-low"
              >
                Low ({summary.low_vif_features ?? 0})
              </Button>
            </div>
          </div>

          {/* VIF Table */}
          <div className="rounded-md border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-semibold">Feature</TableHead>
                  <TableHead className="font-semibold text-right">VIF</TableHead>
                  <TableHead className="font-semibold text-right">Tolerance (1/VIF)</TableHead>
                  <TableHead className="font-semibold text-right">R&sup2; Score</TableHead>
                  <TableHead className="font-semibold text-center">Collinearity Status</TableHead>
                  <TableHead className="font-semibold">Remediation Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredScores.map((row) => (
                  <TableRow key={row.feature} data-testid={`vif-row-${row.feature}`}>
                    <TableCell className="font-medium font-mono text-xs">{row.feature}</TableCell>
                    <TableCell className="text-right font-mono text-xs font-semibold">
                      {row.vif === "Infinity" ? (
                        <span className="text-destructive font-bold">&infin; (Linear Dep.)</span>
                      ) : (
                        row.vif
                      )}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs text-muted-foreground">
                      {row.tolerance}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs text-muted-foreground">
                      {row.r_squared}
                    </TableCell>
                    <TableCell className="text-center">{getSeverityBadge(row.severity)}</TableCell>
                    <TableCell className="text-xs text-muted-foreground max-w-xs">
                      {row.recommendation}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </div>
  );
};

VIFMulticollinearityPanel.propTypes = {
  fileId: PropTypes.string,
};

export default VIFMulticollinearityPanel;
