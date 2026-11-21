/**
 * Analysis Dashboard Component
 * Main container that routes to appropriate analysis panels based on problem type.
 * @module
 */

import { useState, useEffect } from "react";
import axios from "@/shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "@/constants/Urls";
import {
  ConfusionMatrixData,
  RegressionAnalysisData,
  RocCurveData,
} from "@/types/analysis";
import ConfusionMatrixPanel from "./ConfusionMatrixPanel";
import ClassificationReportTable from "./ClassificationReportTable";
import FeatureImportanceChart from "./FeatureImportanceChart";
import PredictionExplorer from "./PredictionExplorer";
import ResidualPlotPanel from "./ResidualPlotPanel";
import RocPrCurvePanel from "./RocPrCurvePanel";
import CalibrationCurvePanel from "./CalibrationCurvePanel";
import InferencePlayground from "./InferencePlayground";
import { useFeatureImportancePoller } from "@/hooks/useFeatureImportancePoller";
import logger from "@/shared/logger";

interface AnalysisDashboardProps {
  jobId: string;
}

type AnalysisType = "classification" | "regression" | "image_classification" | null;

export default function AnalysisDashboard({ jobId }: AnalysisDashboardProps) {
  const [analysisType, setAnalysisType] = useState<AnalysisType>(null);
  const [confusionData, setConfusionData] = useState<ConfusionMatrixData | null>(null);
  const [regressionData, setRegressionData] = useState<RegressionAnalysisData | null>(null);
  const [rocData, setRocData] = useState<RocCurveData | null>(null);
  const [isLoadingRoc, setIsLoadingRoc] = useState<boolean>(false);
  const [rocError, setRocError] = useState<string | null>(null);
  const [isLoadingType, setIsLoadingType] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<"json" | "csv">("json");
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExport = async () => {
    setIsExporting(true);
    setExportError(null);
    try {
      const response = await axios.get(
        `${BACKEND_MODEL_ANALYSIS}/${jobId}/export?format=${exportFormat}`,
        { responseType: "blob" }
      );
      const blob = new Blob([response.data], {
        type: exportFormat === "json" ? "application/json" : "text/csv",
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `analysis-${jobId}.${exportFormat}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      logger.error("Failed to export analysis:", err);
      let parsedDetail: string | null = null;
      if (err.response?.data) {
        try {
          let errorData = err.response.data;
          if (errorData instanceof Blob) {
            const text = await errorData.text();
            errorData = JSON.parse(text);
          } else if (typeof errorData === "string") {
            try {
              errorData = JSON.parse(errorData);
            } catch {
              // ignore
            }
          }
          if (errorData && typeof errorData === "object" && errorData.detail) {
            parsedDetail = errorData.detail;
          }
        } catch {
          // ignore
        }
      }
      setExportError(
        parsedDetail ||
          err.response?.data?.detail ||
          err.message ||
          "Failed to export analysis"
      );
    } finally {
      setIsExporting(false);
    }
  };

  // Feature importance polling
  const {
    data: featureImportanceData,
    isLoading: isLoadingFeatureImportance,
    error: featureImportanceError,
  } = useFeatureImportancePoller(jobId);

  // Determine analysis type by attempting to fetch confusion matrix
  useEffect(() => {
    const determineAnalysisType = async () => {
      setIsLoadingType(true);
      setError(null);

      try {
        // Try confusion matrix first (works for classification & image_classification)
        const response = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/confusion-matrix`);
        const data = response.data as ConfusionMatrixData;
        setAnalysisType(data.analysis_type as AnalysisType);
        setConfusionData(data);
      } catch (err: any) {
        // If 400, it may be a regression model OR the job may not be ready yet
        if (err.response?.status === 400) {
          try {
            // Try residuals endpoint
            const regResponse = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/residuals`);
            const regData = regResponse.data as RegressionAnalysisData;
            setAnalysisType(regData.analysis_type as AnalysisType);
            setRegressionData(regData);
          } catch (regErr: any) {
            logger.error("Failed to fetch regression analysis:", regErr);
            // Preserve the original error detail if available
            const detail = regErr.response?.data?.detail || err.response?.data?.detail;
            setError(detail || regErr.message || err.message || "Failed to load analysis");
          }
        } else {
          logger.error("Failed to fetch confusion matrix:", err);
          setError(err.response?.data?.detail || err.message || "Failed to load analysis");
        }
      } finally {
        setIsLoadingType(false);
      }
    };

    determineAnalysisType();
  }, [jobId]);

  // Fetch ROC & Precision-Recall curves for tabular classification models
  useEffect(() => {
    if (analysisType !== "classification") return;

    const fetchRocData = async () => {
      setIsLoadingRoc(true);
      setRocError(null);
      try {
        const response = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/roc-curve`);
        setRocData(response.data as RocCurveData);
      } catch (err: any) {
        logger.error("Failed to fetch ROC curve analysis:", err);
        setRocError(err.response?.data?.detail || err.message || "Failed to load ROC curves");
      } finally {
        setIsLoadingRoc(false);
      }
    };

    fetchRocData();
  }, [jobId, analysisType]);

  if (isLoadingType) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md">
        <p className="text-sm text-red-800">
          <strong>Error:</strong> {error}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Export Toolbar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
        <div>
          <h2 className="text-lg font-semibold text-gray-800">Model Analysis</h2>
          <p className="text-xs text-gray-500">
            Export diagnostic metrics and summary results for offline reporting
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as "json" | "csv")}
            disabled={isExporting}
            className="text-sm border border-gray-300 rounded-md px-2.5 py-1.5 bg-white text-gray-700 shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            aria-label="Export format"
          >
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
          </select>
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="inline-flex items-center px-3 py-1.5 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isExporting ? "Exporting..." : "Export Analysis"}
          </button>
        </div>
      </div>

      {exportError && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-700">
            <strong>Export Error:</strong> {exportError}
          </p>
        </div>
      )}

      {/* Classification Analysis */}
      {analysisType === "classification" && confusionData && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ConfusionMatrixPanel
              confusionMatrix={confusionData.confusion_matrix}
              classNames={confusionData.class_names}
              accuracy={confusionData.overall_accuracy}
            />
            <ClassificationReportTable
              classificationReport={confusionData.classification_report}
              classNames={confusionData.class_names}
            />
          </div>

          {/* ROC & Precision-Recall Diagnostic Curves */}
          <RocPrCurvePanel
            data={rocData}
            isLoading={isLoadingRoc}
            error={rocError}
          />

          {/* Feature Importance */}
          {isLoadingFeatureImportance && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-md">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                <div>
                  <p className="text-sm text-blue-800 font-medium">
                    Computing feature importance...
                  </p>
                  <p className="text-xs text-blue-600">This may take up to 30 seconds</p>
                </div>
              </div>
            </div>
          )}

          {featureImportanceError && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-800">
                <strong>Feature Importance Error:</strong> {featureImportanceError}
              </p>
            </div>
          )}

          {featureImportanceData && (
            <FeatureImportanceChart
              features={featureImportanceData.features}
              importancesMean={featureImportanceData.importances_mean}
              importancesStd={featureImportanceData.importances_std}
            />
          )}

          {/* Probability Calibration & Reliability Analysis */}
          <CalibrationCurvePanel jobId={jobId} />

          {/* Prediction Explorer */}
          <PredictionExplorer jobId={jobId} />

          {/* Interactive Inference Playground */}
          <InferencePlayground jobId={jobId} defaultExpanded={false} />
        </>
      )}

      {/* Image Classification Analysis */}
      {analysisType === "image_classification" && confusionData && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ConfusionMatrixPanel
              confusionMatrix={confusionData.confusion_matrix}
              classNames={confusionData.class_names}
              accuracy={confusionData.overall_accuracy}
            />
            <ClassificationReportTable
              classificationReport={confusionData.classification_report}
              classNames={confusionData.class_names}
            />
          </div>
        </>
      )}

      {/* Regression Analysis */}
      {analysisType === "regression" && regressionData && (
        <>
          <ResidualPlotPanel
            yPred={regressionData.y_pred}
            yTrue={regressionData.y_true}
            residuals={regressionData.residuals}
            mae={regressionData.mae}
            mse={regressionData.mse}
            rmse={regressionData.rmse}
            r2={regressionData.r2}
          />

          {/* Feature Importance for Regression */}
          {isLoadingFeatureImportance && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-md">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                <div>
                  <p className="text-sm text-blue-800 font-medium">
                    Computing feature importance...
                  </p>
                  <p className="text-xs text-blue-600">This may take up to 30 seconds</p>
                </div>
              </div>
            </div>
          )}

          {featureImportanceError && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-800">
                <strong>Feature Importance Error:</strong> {featureImportanceError}
              </p>
            </div>
          )}

          {featureImportanceData && (
            <FeatureImportanceChart
              features={featureImportanceData.features}
              importancesMean={featureImportanceData.importances_mean}
              importancesStd={featureImportanceData.importances_std}
            />
          )}

          {/* Interactive Inference Playground */}
          <InferencePlayground jobId={jobId} defaultExpanded={false} />
        </>
      )}
    </div>
  );
}
