import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import {
  Sparkles,
  Filter,
  CheckSquare,
  Square,
  AlertTriangle,
  Info,
  Loader2,
  Trash2,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  analyzeFeatureRelevance,
  applyFeatureSelection,
} from "../../services/FeatureSelectionService";
import { getCovMatrix } from "../../services/FileServices";
import FeedbackDialog from "../shared/FeedbackDialog";
import logger from "../../shared/logger";

/**
 * Feature Relevance & Automated Selection Studio.
 *
 * Computes Mutual Information, ANOVA F-statistic, variance, and composite scores
 * for all candidate predictive features against a target, offering one-click
 * pruning of uninformative and zero-variance columns.
 */
export default function FeatureSelectionStudio({ fileId, onPruned }) {
  const [columns, setColumns] = useState([]);
  const [targetField, setTargetField] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPruning, setIsPruning] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState(new Set());
  const [error, setError] = useState(null);
  const [feedback, setFeedback] = useState({ open: false, success: true, message: "" });
  const [confirmModalOpen, setConfirmModalOpen] = useState(false);

  // Fetch available columns for target selection
  useEffect(() => {
    let isMounted = true;
    const fetchColumns = async () => {
      try {
        const response = await getCovMatrix(fileId);
        if (isMounted && response?.data_types) {
          const cols = Object.keys(response.data_types);
          setColumns(cols);
          if (cols.length > 0) {
            setTargetField((prev) => prev || cols[cols.length - 1]);
          }
        }
      } catch (err) {
        logger.error("Error loading column list for feature selection:", err);
      }
    };
    if (fileId) {
      fetchColumns();
    }
    return () => {
      isMounted = false;
    };
  }, [fileId]);

  // Execute feature relevance analysis
  const handleAnalyze = async () => {
    if (!targetField) {
      setError("Please select a target column to evaluate features against.");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await analyzeFeatureRelevance(fileId, { target_field: targetField });
      setAnalysisResult(data);

      // Default selection: select all features that are not CONSTANT_OR_REDUNDANT
      const initialSelected = new Set(
        data.features
          .filter((f) => f.tier !== "CONSTANT_OR_REDUNDANT" && !f.is_constant)
          .map((f) => f.feature),
      );
      // Fallback: if all were pruned, select the top 1
      if (initialSelected.size === 0 && data.features.length > 0) {
        initialSelected.add(data.features[0].feature);
      }
      setSelectedFeatures(initialSelected);
    } catch (err) {
      const msg =
        err.response?.data?.message ||
        err.response?.data?.detail ||
        err.message ||
        "Failed to analyze feature relevance";
      setError(msg);
      logger.error("Analysis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Toggle individual feature selection
  const handleToggleFeature = (featureName) => {
    setSelectedFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(featureName)) {
        next.delete(featureName);
      } else {
        next.add(featureName);
      }
      return next;
    });
  };

  // Quick Preset Handlers
  const handleSelectAll = () => {
    if (!analysisResult) return;
    setSelectedFeatures(new Set(analysisResult.features.map((f) => f.feature)));
  };

  const handleClearAll = () => {
    setSelectedFeatures(new Set());
  };

  const handleSelectTopK = (k) => {
    if (!analysisResult) return;
    const topK = analysisResult.features.slice(0, k).map((f) => f.feature);
    setSelectedFeatures(new Set(topK));
  };

  const handleSelectInformativeOnly = () => {
    if (!analysisResult) return;
    const informative = analysisResult.features
      .filter((f) => f.tier === "CRITICAL" || f.tier === "INFORMATIVE")
      .map((f) => f.feature);
    setSelectedFeatures(new Set(informative));
  };

  // Apply pruning
  const handleConfirmPrune = async () => {
    if (selectedFeatures.size === 0) {
      setError("Cannot prune all features. Please select at least one feature to retain.");
      return;
    }
    setIsPruning(true);
    setError(null);
    try {
      const result = await applyFeatureSelection(fileId, {
        selected_features: Array.from(selectedFeatures),
      });
      setFeedback({
        open: true,
        success: true,
        message: result.message || "Dataset successfully pruned with selected features.",
      });
      setConfirmModalOpen(false);
      // Trigger parent callback to refresh dataset preview and stats
      if (onPruned) onPruned();
      // Re-run analysis on pruned dataset
      handleAnalyze();
    } catch (err) {
      const msg =
        err.response?.data?.message ||
        err.response?.data?.detail ||
        err.message ||
        "Failed to prune dataset";
      setFeedback({
        open: true,
        success: false,
        message: msg,
      });
      logger.error("Pruning error:", err);
    } finally {
      setIsPruning(false);
    }
  };

  const getTierBadge = (tier, isConstant) => {
    if (isConstant || tier === "CONSTANT_OR_REDUNDANT") {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200">
          Redundant / Constant
        </span>
      );
    }
    if (tier === "CRITICAL") {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
          Critical
        </span>
      );
    }
    if (tier === "INFORMATIVE") {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-800 border border-blue-200">
          Informative
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">
        Weak
      </span>
    );
  };

  return (
    <Card className="border border-border/80 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="text-lg font-bold flex items-center gap-2 text-foreground">
              <Sparkles className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
              Feature Relevance & Selection Studio
            </CardTitle>
            <CardDescription className="text-xs text-muted-foreground">
              Score feature-to-target dependency via Mutual Information, ANOVA F-statistics, and
              variance. Prune redundant columns to reduce overfitting.
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 pt-1">
        <FeedbackDialog
          open={feedback.open}
          onClose={() => setFeedback((prev) => ({ ...prev, open: false }))}
          success={feedback.success}
          message={feedback.message}
        />

        {/* Configuration Bar */}
        <div className="flex flex-wrap items-center gap-3 p-3 bg-muted/40 rounded-lg border border-border/60">
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="fs-target-select" className="font-medium text-foreground">
              Target Column:
            </label>
            <select
              id="fs-target-select"
              value={targetField}
              onChange={(e) => setTargetField(e.target.value)}
              className="px-2.5 py-1.5 text-sm border rounded-md bg-background focus:ring-2 focus:ring-primary/20"
            >
              <option value="" disabled>
                Choose target
              </option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !targetField}
            size="sm"
            className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <TrendingUp className="w-3.5 h-3.5" />
                Analyze Relevance
              </>
            )}
          </Button>

          {analysisResult && (
            <span className="text-xs text-muted-foreground ml-auto">
              Evaluated {analysisResult.total_samples} samples ({analysisResult.problem_type})
            </span>
          )}
        </div>

        {error && (
          <div className="flex items-start gap-2 p-3 text-sm rounded-md bg-destructive/10 text-destructive border border-destructive/20">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Analysis Results View */}
        {analysisResult && (
          <div className="space-y-4">
            {/* Stat Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-background rounded-lg border border-border shadow-2xs">
                <p className="text-xs text-muted-foreground font-medium">Candidate Features</p>
                <p className="text-xl font-bold text-foreground mt-1">
                  {analysisResult.total_features}
                </p>
              </div>
              <div className="p-3 bg-emerald-50/50 dark:bg-emerald-950/20 rounded-lg border border-emerald-200 dark:border-emerald-800/40">
                <p className="text-xs text-emerald-700 dark:text-emerald-400 font-medium">
                  Critical & Informative
                </p>
                <p className="text-xl font-bold text-emerald-800 dark:text-emerald-300 mt-1">
                  {analysisResult.critical_features_count +
                    analysisResult.informative_features_count}
                </p>
              </div>
              <div className="p-3 bg-amber-50/50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-800/40">
                <p className="text-xs text-amber-700 dark:text-amber-400 font-medium">
                  Weak Candidates
                </p>
                <p className="text-xl font-bold text-amber-800 dark:text-amber-300 mt-1">
                  {analysisResult.weak_features_count}
                </p>
              </div>
              <div className="p-3 bg-rose-50/50 dark:bg-rose-950/20 rounded-lg border border-rose-200 dark:border-rose-800/40">
                <p className="text-xs text-rose-700 dark:text-rose-400 font-medium">
                  Constant / Redundant
                </p>
                <p className="text-xl font-bold text-rose-800 dark:text-rose-300 mt-1">
                  {analysisResult.constant_features_count}
                </p>
              </div>
            </div>

            {/* Presets and Action Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-muted/30 rounded-lg border border-border/60">
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <span className="font-semibold text-foreground flex items-center gap-1 mr-1">
                  <Filter className="w-3.5 h-3.5 text-muted-foreground" />
                  Presets:
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={() => handleSelectTopK(3)}
                >
                  Top 3
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={() => handleSelectTopK(5)}
                >
                  Top 5
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={() => handleSelectTopK(10)}
                >
                  Top 10
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={handleSelectInformativeOnly}
                >
                  Informative Only
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={handleSelectAll}
                >
                  Select All
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={handleClearAll}
                >
                  Clear
                </Button>
              </div>

              <div className="flex items-center gap-3">
                <span
                  data-testid="selection-summary"
                  className="text-xs font-medium text-foreground"
                >
                  <span className="font-bold text-primary">{selectedFeatures.size}</span> of{" "}
                  {analysisResult.total_features} selected (
                  {analysisResult.total_features - selectedFeatures.size} pruned)
                </span>
                <Button
                  onClick={() => setConfirmModalOpen(true)}
                  disabled={selectedFeatures.size === 0 || isPruning}
                  size="sm"
                  variant="destructive"
                  className="flex items-center gap-1.5 text-xs h-8"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Prune Dataset
                </Button>
              </div>
            </div>

            {/* Ranked Table */}
            <div className="overflow-x-auto border border-border rounded-lg bg-card">
              <table className="w-full text-xs text-left">
                <thead className="bg-muted/60 text-muted-foreground font-semibold border-b border-border">
                  <tr>
                    <th className="py-2.5 px-3 w-10 text-center">Select</th>
                    <th className="py-2.5 px-3">Feature Column</th>
                    <th className="py-2.5 px-3">Relevance Tier</th>
                    <th className="py-2.5 px-3 w-48">Importance Score</th>
                    <th className="py-2.5 px-3 text-right">Mutual Info</th>
                    <th className="py-2.5 px-3 text-right">F-Statistic</th>
                    <th className="py-2.5 px-3 text-right">Variance</th>
                    <th className="py-2.5 px-3 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {analysisResult.features.map((feat) => {
                    const isSelected = selectedFeatures.has(feat.feature);
                    return (
                      <tr
                        key={feat.feature}
                        className={`hover:bg-muted/40 transition-colors ${
                          !isSelected ? "opacity-60 bg-muted/20" : ""
                        }`}
                      >
                        <td className="py-2 px-3 text-center">
                          <button
                            type="button"
                            onClick={() => handleToggleFeature(feat.feature)}
                            className="text-primary hover:text-primary/80 focus:outline-hidden"
                            aria-label={`Toggle selection for ${feat.feature}`}
                          >
                            {isSelected ? (
                              <CheckSquare className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                            ) : (
                              <Square className="w-4 h-4 text-muted-foreground" />
                            )}
                          </button>
                        </td>
                        <td className="py-2 px-3 font-medium text-foreground">{feat.feature}</td>
                        <td className="py-2 px-3">{getTierBadge(feat.tier, feat.is_constant)}</td>
                        <td className="py-2 px-3">
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-muted rounded-full h-2 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  feat.normalized_score >= 60
                                    ? "bg-emerald-500"
                                    : feat.normalized_score >= 25
                                      ? "bg-blue-500"
                                      : "bg-amber-500"
                                }`}
                                style={{
                                  width: `${Math.min(100, Math.max(2, feat.normalized_score))}%`,
                                }}
                              />
                            </div>
                            <span className="font-mono text-xs text-muted-foreground">
                              {feat.normalized_score.toFixed(1)}
                            </span>
                          </div>
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-foreground">
                          {feat.mutual_info.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-foreground">
                          {feat.f_statistic !== null ? feat.f_statistic.toFixed(2) : "-"}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-muted-foreground">
                          {feat.variance.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 text-center">
                          <span
                            className={`px-1.5 py-0.5 text-[10px] font-medium rounded uppercase ${
                              feat.suggested_action === "keep"
                                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                                : feat.suggested_action === "review"
                                  ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                                  : "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300"
                            }`}
                          >
                            {feat.suggested_action}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Confirmation Modal */}
        {confirmModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="bg-background rounded-lg border border-border shadow-xl max-w-md w-full p-5 space-y-4 animate-in fade-in zoom-in-95">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-destructive/10 text-destructive rounded-full">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-foreground">Confirm Dataset Pruning</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                You are about to retain <strong>{selectedFeatures.size}</strong> feature(s) and
                prune <strong>{analysisResult.total_features - selectedFeatures.size}</strong>{" "}
                unselected feature(s).
              </p>
              <div className="p-3 bg-muted/40 rounded border border-border text-xs text-muted-foreground space-y-1">
                <div className="flex items-center gap-1 text-foreground font-semibold">
                  <Info className="w-3.5 h-3.5 text-blue-500" /> Safe Modification:
                </div>
                <p data-testid="backup-note">
                  A <code>.bak</code> backup file will be created. The target column{" "}
                  <strong>{targetField}</strong> will remain preserved.
                </p>
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setConfirmModalOpen(false)}
                  disabled={isPruning}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleConfirmPrune}
                  disabled={isPruning}
                  className="flex items-center gap-1.5"
                >
                  {isPruning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                  Confirm & Prune
                </Button>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

FeatureSelectionStudio.propTypes = {
  fileId: PropTypes.string.isRequired,
  onPruned: PropTypes.func,
};
