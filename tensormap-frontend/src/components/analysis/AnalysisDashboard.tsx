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
} from "@/types/analysis";
import ConfusionMatrixPanel from "./ConfusionMatrixPanel";
import ClassificationReportTable from "./ClassificationReportTable";
import FeatureImportanceChart from "./FeatureImportanceChart";
import PredictionExplorer from "./PredictionExplorer";
import ResidualPlotPanel from "./ResidualPlotPanel";
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
  const [isLoadingType, setIsLoadingType] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

          {/* Feature Importance */}
          {isLoadingFeatureImportance && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-md">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                <div>
                  <p className="text-sm text-blue-800 font-medium">Computing feature importance...</p>
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

          {/* Prediction Explorer */}
          <PredictionExplorer jobId={jobId} />
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
                  <p className="text-sm text-blue-800 font-medium">Computing feature importance...</p>
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
        </>
      )}
    </div>
  );
}
