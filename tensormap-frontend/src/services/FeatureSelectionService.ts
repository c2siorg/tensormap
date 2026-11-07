import axios from "axios";
import http from "../shared/Axios";
import * as urls from "../constants/Urls";

const endpoint = `${urls.BACKEND_FEATURE_SELECTION}/`;

const getErrorMessage = (error: unknown, fallback: string) => {
  if (axios.isAxiosError(error)) {
    const message = error.response?.data?.message;
    if (typeof message === "string") return message;
  }
  return error instanceof Error ? error.message : fallback;
};

export type FeatureScore = {
  feature: string;
  mutual_information: number;
  anova_f: number;
  variance: number;
  composite_score: number;
};

export type FeatureAnalysis = {
  target: string;
  task: "classification" | "regression";
  sample_rows: number;
  scores: FeatureScore[];
};

export const analyzeFeatures = async (
  fileId: string,
  target: string,
  task: "auto" | "classification" | "regression",
): Promise<FeatureAnalysis> => {
  try {
    const response = await http.post(`${endpoint}${fileId}/analyze`, { target, task });
    if (!response.data.success) throw new Error(response.data.message || "Feature analysis failed");
    return response.data.data as FeatureAnalysis;
  } catch (error) {
    throw new Error(getErrorMessage(error, "Feature analysis failed"));
  }
};

export const applyFeatureSelection = async (
  fileId: string,
  target: string,
  columnsToDrop: string[],
) => {
  try {
    const response = await http.post(`${endpoint}${fileId}/apply`, {
      target,
      columns_to_drop: columnsToDrop,
    });
    if (!response.data.success)
      throw new Error(response.data.message || "Could not apply feature selection");
    return response.data;
  } catch (error) {
    throw new Error(getErrorMessage(error, "Could not apply feature selection"));
  }
};
