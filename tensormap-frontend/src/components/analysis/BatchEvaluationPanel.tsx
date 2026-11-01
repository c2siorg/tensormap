/**
 * Batch Evaluation & Holdout Dataset Testing Panel
 * Evaluates trained models against external test datasets, computes comprehensive
 * metric scorecards, displays top prediction errors, and exports batch predictions.
 * @module
 */

import React, { useState, useEffect } from "react";
import {
  FileText,
  Play,
  Download,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  TrendingUp,
  BarChart2,
} from "lucide-react";
import { getAllFiles } from "../../services/FileServices";
import {
  evaluateHoldoutDataset,
  downloadBatchPredictionsCsv,
} from "../../services/EvaluationService";
import { BatchEvaluationResponse, FileItem } from "../../types/evaluation";
import logger from "../../shared/logger";

interface BatchEvaluationPanelProps {
  jobId: string;
  problemType: "classification" | "regression" | "image_classification" | null;
  projectId?: string;
}

export default function BatchEvaluationPanel({
  jobId,
  problemType,
  projectId,
}: BatchEvaluationPanelProps) {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<string>("");
  const [isLoadingFiles, setIsLoadingFiles] = useState<boolean>(true);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [evaluationResult, setEvaluationResult] = useState<BatchEvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showErrorDetails, setShowErrorDetails] = useState<boolean>(true);

  // Fetch available CSV datasets
  useEffect(() => {
    let isMounted = true;
    setIsLoadingFiles(true);

    getAllFiles(projectId)
      .then((data: any[]) => {
        if (!isMounted) return;
        const csvFiles = (data || []).filter(
          (f) => f.file_type?.toLowerCase() === "csv" || f.file_name?.endsWith(".csv"),
        );
        setFiles(csvFiles);
        if (csvFiles.length > 0) {
          setSelectedFileId(csvFiles[0].file_id);
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        logger.error("Failed to load project files for evaluation:", err);
      })
      .finally(() => {
        if (isMounted) setIsLoadingFiles(false);
      });

    return () => {
      isMounted = false;
    };
  }, [projectId]);

  const handleRunEvaluation = async () => {
    if (!selectedFileId) {
      setError("Please select a dataset file to evaluate.");
      return;
    }

    setIsEvaluating(true);
    setError(null);

    try {
      const result = await evaluateHoldoutDataset(jobId, selectedFileId);
      setEvaluationResult(result);
    } catch (err: any) {
      logger.error("Holdout evaluation failed:", err);
      const msg =
        err.response?.data?.message ||
        err.response?.data?.detail ||
        err.message ||
        "Evaluation failed on the selected dataset.";
      setError(msg);
      setEvaluationResult(null);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleDownloadPredictions = async () => {
    if (!selectedFileId) {
      setError("Please select a dataset file to generate predictions.");
      return;
    }

    setIsExporting(true);
    setError(null);

    try {
      await downloadBatchPredictionsCsv(jobId, selectedFileId);
    } catch (err: any) {
      logger.error("Batch predictions export failed:", err);
      const msg =
        err.response?.data?.message ||
        err.response?.data?.detail ||
        err.message ||
        "Failed to export batch predictions.";
      setError(msg);
    } finally {
      setIsExporting(false);
    }
  };

  const isClassification = problemType === "classification";
  const isRegression = problemType === "regression";
  const metrics = evaluationResult?.metrics || {};

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6 space-y-6">
      {/* Header */}
      <div className="border-b border-gray-100 pb-4">
        <div className="flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            Holdout Dataset Evaluation & Batch Scoring
          </h3>
        </div>
        <p className="text-sm text-gray-500 mt-1">
          Evaluate trained model accuracy against external holdout test datasets, inspect
          misclassified and high-residual samples, and export batch predictions as CSV.
        </p>
      </div>

      {/* Dataset Selection Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
        <div className="md:col-span-2 space-y-1.5">
          <label htmlFor="eval-file-select" className="block text-sm font-medium text-gray-700">
            Select Test Dataset
          </label>
          <div className="relative">
            <select
              id="eval-file-select"
              aria-label="Select Test Dataset"
              value={selectedFileId}
              onChange={(e) => {
                setSelectedFileId(e.target.value);
                setEvaluationResult(null);
                setError(null);
              }}
              disabled={isLoadingFiles || isEvaluating || isExporting || files.length === 0}
              className="w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            >
              {files.length === 0 ? (
                <option value="">
                  {isLoadingFiles ? "Loading datasets..." : "No CSV datasets found"}
                </option>
              ) : (
                files.map((file) => (
                  <option key={file.file_id} value={file.file_id}>
                    {file.file_name}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRunEvaluation}
            disabled={!selectedFileId || isEvaluating || isExporting}
            className="flex-1 inline-flex items-center justify-center gap-1.5 px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-colors"
          >
            {isEvaluating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Evaluating...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Evaluate Dataset</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleDownloadPredictions}
            disabled={!selectedFileId || isEvaluating || isExporting}
            title="Download batch predictions as CSV"
            className="inline-flex items-center justify-center gap-1.5 px-3 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-colors"
          >
            {isExporting ? (
              <RefreshCw className="w-4 h-4 animate-spin text-gray-500" />
            ) : (
              <Download className="w-4 h-4 text-gray-600" />
            )}
            <span className="hidden sm:inline">Scored CSV</span>
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div
          role="alert"
          className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-md text-sm text-red-800"
        >
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold">Evaluation Error</p>
            <p className="mt-0.5 text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Evaluation Results Card */}
      {evaluationResult && (
        <div className="space-y-6 pt-2">
          {/* Summary Metric Cards */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wider flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                <span>Evaluation Scorecard ({evaluationResult.total_samples} samples)</span>
              </h4>
              <span className="text-xs text-gray-400">
                {new Date(evaluationResult.evaluated_at).toLocaleTimeString()}
              </span>
            </div>

            {isClassification ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                <div className="p-3 bg-blue-50/60 border border-blue-100 rounded-md">
                  <span className="text-xs font-medium text-blue-700 uppercase">Accuracy</span>
                  <div className="text-xl font-bold text-blue-900 mt-1">
                    {(metrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="p-3 bg-emerald-50/60 border border-emerald-100 rounded-md">
                  <span className="text-xs font-medium text-emerald-700 uppercase">Macro F1</span>
                  <div className="text-xl font-bold text-emerald-900 mt-1">
                    {metrics.f1_macro?.toFixed(4) ?? "-"}
                  </div>
                </div>

                <div className="p-3 bg-purple-50/60 border border-purple-100 rounded-md">
                  <span className="text-xs font-medium text-purple-700 uppercase">Weighted F1</span>
                  <div className="text-xl font-bold text-purple-900 mt-1">
                    {metrics.f1_weighted?.toFixed(4) ?? "-"}
                  </div>
                </div>

                <div className="p-3 bg-indigo-50/60 border border-indigo-100 rounded-md">
                  <span className="text-xs font-medium text-indigo-700 uppercase">Precision</span>
                  <div className="text-xl font-bold text-indigo-900 mt-1">
                    {metrics.precision_macro?.toFixed(4) ?? "-"}
                  </div>
                </div>

                <div className="p-3 bg-amber-50/60 border border-amber-100 rounded-md">
                  <span className="text-xs font-medium text-amber-700 uppercase">Log Loss</span>
                  <div className="text-xl font-bold text-amber-900 mt-1">
                    {metrics.log_loss != null ? metrics.log_loss.toFixed(4) : "N/A"}
                  </div>
                </div>
              </div>
            ) : isRegression ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                <div className="p-3 bg-emerald-50/60 border border-emerald-100 rounded-md">
                  <span className="text-xs font-medium text-emerald-700 uppercase">RMSE</span>
                  <div className="text-xl font-bold text-emerald-900 mt-1">
                    {metrics.rmse?.toFixed(4) ?? "-"}
                  </div>
                </div>

                <div className="p-3 bg-blue-50/60 border border-blue-100 rounded-md">
                  <span className="text-xs font-medium text-blue-700 uppercase">MAE</span>
                  <div className="text-xl font-bold text-blue-900 mt-1">
                    {metrics.mae?.toFixed(4) ?? "-"}
                  </div>
                </div>

                <div className="p-3 bg-amber-50/60 border border-amber-100 rounded-md">
                  <span className="text-xs font-medium text-amber-700 uppercase">R² Score</span>
                  <div className="text-xl font-bold text-amber-900 mt-1">
                    {metrics.r2 != null ? metrics.r2.toFixed(4) : "N/A"}
                  </div>
                </div>

                <div className="p-3 bg-purple-50/60 border border-purple-100 rounded-md">
                  <span className="text-xs font-medium text-purple-700 uppercase">MSE</span>
                  <div className="text-xl font-bold text-purple-900 mt-1">
                    {metrics.mse?.toFixed(4) ?? "-"}
                  </div>
                </div>

                <div className="p-3 bg-rose-50/60 border border-rose-100 rounded-md">
                  <span className="text-xs font-medium text-rose-700 uppercase">Max Error</span>
                  <div className="text-xl font-bold text-rose-900 mt-1">
                    {metrics.max_error?.toFixed(4) ?? "-"}
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          {/* Top Prediction Errors Inspector */}
          {evaluationResult.top_errors && evaluationResult.top_errors.length > 0 ? (
            <div className="border border-gray-200 rounded-md overflow-hidden">
              <button
                type="button"
                onClick={() => setShowErrorDetails(!showErrorDetails)}
                className="w-full flex items-center justify-between p-3.5 bg-gray-50 text-left hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    Top Prediction Errors ({evaluationResult.top_errors.length})
                  </span>
                </div>
                {showErrorDetails ? (
                  <ChevronUp className="w-4 h-4 text-gray-500" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-gray-500" />
                )}
              </button>

              {showErrorDetails && (
                <div className="overflow-x-auto">
                  <table
                    className="w-full text-xs text-left border-collapse"
                    aria-label="Top prediction errors"
                  >
                    <thead className="bg-gray-100/70 text-gray-600 uppercase border-b border-gray-200">
                      <tr>
                        <th scope="col" className="px-3.5 py-2.5 font-medium">
                          Row
                        </th>
                        <th scope="col" className="px-3.5 py-2.5 font-medium">
                          Actual
                        </th>
                        <th scope="col" className="px-3.5 py-2.5 font-medium">
                          Predicted
                        </th>
                        <th scope="col" className="px-3.5 py-2.5 font-medium">
                          {isClassification ? "Confidence" : "Residual"}
                        </th>
                        <th scope="col" className="px-3.5 py-2.5 font-medium">
                          Feature Inputs
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {evaluationResult.top_errors.map((errItem, idx) => (
                        <tr key={idx} className="hover:bg-gray-50/50">
                          <td className="px-3.5 py-2.5 font-mono text-gray-500">
                            #{errItem.index}
                          </td>
                          <td className="px-3.5 py-2.5 font-semibold text-gray-900">
                            {errItem.actual_label}
                          </td>
                          <td className="px-3.5 py-2.5 font-semibold text-red-600">
                            {errItem.predicted_label}
                          </td>
                          <td className="px-3.5 py-2.5">
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-red-100 text-red-800">
                              {isClassification && errItem.confidence != null
                                ? `${(errItem.confidence * 100).toFixed(1)}%`
                                : errItem.loss_or_residual.toFixed(4)}
                            </span>
                          </td>
                          <td className="px-3.5 py-2.5 text-gray-500 font-mono text-[11px]">
                            {Object.entries(errItem.features)
                              .slice(0, 4)
                              .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(2) : v}`)
                              .join(", ")}
                            {Object.keys(errItem.features).length > 4 && " ..."}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <div className="p-4 bg-green-50 border border-green-200 rounded-md text-sm text-green-800 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-600" />
              <span>
                Perfect evaluation score! No misclassified or high-error samples detected.
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
