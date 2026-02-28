import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import * as strings from "../constants/Strings";
import logger from "../shared/logger";
import { BACKEND_GET_LAYERS } from "../constants/Urls";

/**
 * Sends a model graph and training config to the backend for validation.
 *
 * On network error, returns a fallback `{ success: false }` object
 * instead of throwing, so callers always receive a response shape.
 *
 * @param {object} data - Payload with `code`, `model`, and `project_id`.
 * @returns {Promise<{ success: boolean, message: string }>}
 */
export const validateModel = async (data) =>
  axios
    .post(urls.BACKEND_VALIDATE_MODEL, data)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occured" };
    });

/**
 * Fetches the list of validated model names, optionally scoped to a project.
 *
 * @param {string} [projectId]
 * @returns {Promise<string[]>} Array of model name strings.
 */
export const getAllModels = async (projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  return axios
    .get(urls.BACKEND_GET_ALL_MODELS, { params })
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.data;
      }
      return [];
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};

/**
 * Downloads the generated Python training script for a model.
 *
 * Creates a temporary download link, triggers the browser download,
 * and cleans up the object URL afterwards.
 *
 * @param {string} model_name
 * @param {string} [projectId]
 * @returns {Promise<void>}
 */
export const download_code = async (model_name, projectId) => {
  const data = { model_name, ...(projectId && { project_id: projectId }) };
  return axios
    .post(urls.BACKEND_DOWNLOAD_CODE, data)
    .then((resp) => {
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(
        new Blob([resp.data], { type: "application/octet-stream" }),
      );
      link.download = data.model_name + strings.MODEL_EXTENSION;

      document.body.appendChild(link);

      link.click();

      setTimeout(() => {
        window.URL.revokeObjectURL(link.href);
        document.body.removeChild(link);
      }, 200);
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};

/**
 * Starts a training run for a validated model on the backend.
 *
 * Training progress is delivered separately via the Socket.IO
 * `/dl-result` namespace.
 *
 * @param {string} modelName
 * @param {string} [projectId]
 * @returns {Promise<string>} Success message from the API.
 */
export const saveModel = async (data) =>
  axios
    .post(urls.BACKEND_SAVE_MODEL, data)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occurred" };
    });

export const updateTrainingConfig = async (data) =>
  axios
    .patch(urls.BACKEND_UPDATE_TRAINING_CONFIG, data)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occurred" };
    });

/**
 * Fetches the full ReactFlow graph for a saved model.
 *
 * @param {string} modelName
 * @param {string} [projectId]
 * @returns {Promise<{ success: boolean, data?: { model_name: string, graph: object } }>}
 */
export const getModelGraph = async (modelName, projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  return axios
    .get(`${urls.BACKEND_GET_MODEL_GRAPH}/${encodeURIComponent(modelName)}/graph`, { params })
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      return { success: false };
    });
};

export const runModel = async (modelName, projectId) => {
  const data = {
    model_name: modelName,
    ...(projectId && { project_id: projectId }),
  };

  return axios
    .post(urls.BACKEND_RUN_MODEL, data)
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.message;
      }
      throw new Error(resp.data);
    })
    .catch((err) => {
      throw err;
    });
};

export const getLayerRegistry = async () => {
  try {
    // Explicitly point to the backend port 4300
    const response = await fetch(`http://127.0.0.1:4300/api/v1/layers`);
    const json = await response.json();
    return json.data.layers;
  } catch (error) {
    console.error("Failed to fetch layer registry:", error);
    return {};
  }
};