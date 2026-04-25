import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import * as strings from "../constants/Strings";
import logger from "../shared/logger";

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
 * Fetches the list of validated model objects, optionally scoped to a project.
 *
 * @param {string} [projectId]
 * @returns {Promise<Array<{ id: number, model_name: string }>>} Array of model objects.
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
 * Fetches the enriched training history for models, optionally scoped to a project.
 *
 * @param {string} [projectId]
 * @returns {Promise<Array<object>>} Array of enriched model objects.
 */
export const getTrainingHistory = async (projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  return axios
    .get(urls.BACKEND_GET_TRAINING_HISTORY, { params })
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.data;
      }
      return [];
    })
    .catch(() => []);
};

/**
 * Get interpretability metrics for a model.
 */
export const getModelInterpret = async (modelName, fileId, projectId) => {
  const params = new URLSearchParams();
  if (fileId) params.append("file_id", fileId);
  if (projectId) params.append("project_id", projectId);

  return axios
    .get(`${urls.BACKEND_MODEL_INTERPRET}/${modelName}?${params}`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      return { success: false, message: "Failed to get interpretability" };
    });
};

/**
 * Deletes a saved model by its database ID.
 *
 * @param {number} modelId
 * @returns {Promise<{ success: boolean, message: string }>}
 */
export const deleteModel = async (modelId) =>
  axios
    .delete(`${urls.BACKEND_DELETE_MODEL}/${modelId}`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occurred" };
    });

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

/**
 * Export a trained model in the specified format.
 *
 * @param {string} model_name - Name of model to export
 * @param {string} format - Format: 'savedmodel', 'tflite', or 'onnx'
 * @param {string} [projectId] - Optional project ID
 * @returns {Promise<void>}
 */
export const exportModel = async (model_name, format, projectId) => {
  const params = new URLSearchParams({ format });
  if (projectId) params.append("project_id", projectId);

  return axios
    .get(`${urls.BACKEND_MODEL_EXPORT}/${model_name}?${params}`, { responseType: "blob" })
    .then((resp) => {
      if (resp.status === 200) {
        let extension = format;
        if (format === "savedmodel") extension = "tar.gz";

        const link = document.createElement("a");
        link.href = window.URL.createObjectURL(new Blob([resp.data]));
        link.download = `${model_name}.${extension}`;
        document.body.appendChild(link);
        link.click();
        setTimeout(() => {
          window.URL.revokeObjectURL(link.href);
          document.body.removeChild(link);
        }, 200);
      } else {
        throw new Error("Export failed");
      }
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};
