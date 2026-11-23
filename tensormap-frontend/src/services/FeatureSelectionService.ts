import axios from "../shared/Axios";
import { BACKEND_FEATURE_SELECTION } from "../constants/Urls";
import type {
  ApplyFeatureSelectionRequest,
  ApplyFeatureSelectionResponse,
  FeatureSelectionAnalyzeRequest,
  FeatureSelectionAnalyzeResponse,
  RestoreFeatureSelectionResponse,
} from "../types/featureSelection";
import logger from "../shared/logger";

/**
 * Runs statistical and information-theoretic feature relevance analysis on a dataset.
 *
 * @param fileId - Dataset file UUID
 * @param payload - Optional configuration (target field, problem type override)
 * @returns Feature relevance analysis including MI, F-statistics, and ranked tiers
 */
export const analyzeFeatureRelevance = async (
  fileId: string,
  payload: FeatureSelectionAnalyzeRequest = {},
): Promise<FeatureSelectionAnalyzeResponse> => {
  try {
    const response = await axios.post<FeatureSelectionAnalyzeResponse>(
      `${BACKEND_FEATURE_SELECTION}/${fileId}/analyze`,
      payload,
    );
    return response.data;
  } catch (error) {
    logger.error("Error analyzing feature relevance:", error);
    throw error;
  }
};

/**
 * Prunes a dataset CSV down to selected features plus target, creating a restorable .bak copy.
 *
 * @param fileId - Dataset file UUID
 * @param payload - Selection criteria (explicit selected_features, top_k, or min_score)
 * @returns Summary of retained and dropped features
 */
export const applyFeatureSelection = async (
  fileId: string,
  payload: ApplyFeatureSelectionRequest,
): Promise<ApplyFeatureSelectionResponse> => {
  try {
    const response = await axios.post<ApplyFeatureSelectionResponse>(
      `${BACKEND_FEATURE_SELECTION}/${fileId}/apply`,
      payload,
    );
    return response.data;
  } catch (error) {
    logger.error("Error applying feature selection:", error);
    throw error;
  }
};

/**
 * Restores a dataset CSV back to its original pre-pruned state from .bak backup.
 *
 * @param fileId - Dataset file UUID
 * @returns Summary of restored dataset
 */
export const restoreFeatureSelectionBackup = async (
  fileId: string,
): Promise<RestoreFeatureSelectionResponse> => {
  try {
    const response = await axios.post<RestoreFeatureSelectionResponse>(
      `${BACKEND_FEATURE_SELECTION}/${fileId}/restore`,
    );
    return response.data;
  } catch (error) {
    logger.error("Error restoring feature selection backup:", error);
    throw error;
  }
};
