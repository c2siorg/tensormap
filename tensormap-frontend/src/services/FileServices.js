import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import logger from "../shared/logger";

/**
 * Fetches the list of uploaded files, optionally scoped to a project.
 *
 * @param {string} [projectId] - Project ID to filter by.
 * @returns {Promise<Array>} Array of file objects from the API.
 */
export const getAllFiles = async (projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  return axios
    .get(urls.BACKEND_GET_ALL_FILES, { params })
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
 * Uploads a CSV file to the backend.
 *
 * @param {File} file - Browser File object to upload.
 * @param {string} [projectId] - Project to associate the file with.
 * @returns {Promise<boolean>} `true` on success.
 */
export const uploadFile = async (file, projectId) => {
  const formData = new FormData();
  formData.append("data", file);
  if (projectId) {
    formData.append("project_id", projectId);
  }

  return axios
    .post(urls.BACKEND_FILE_UPLOAD, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((resp) => {
      if (resp.data.success === true) {
        return true;
      }
      return false;
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};

/**
 * Fetches the correlation matrix, data types, and metrics for a file.
 *
 * @param {string} file_id - ID of the uploaded file.
 * @returns {Promise<object>} Object with correlation_matrix, data_types, and metric.
 */
export const getCovMatrix = async (file_id) =>
  axios
    .get(urls.BACKEND_GET_COV_MATRIX + file_id)
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.data;
      }
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });

/**
 * Fetches the raw data content of a file as a JSON string or paginated object.
 *
 * @param {string} file_id - ID of the uploaded file.
 * @param {number} page - Page number.
 * @param {number} pageSize - Number of rows per page.
 * @returns {Promise<object>} Response data containing data array and pagination metadata.
 */
export const getFileData = async (file_id, page = 1, pageSize = 50) => {
  try {
    const params = { page, page_size: pageSize };
    const response = await axios.get(urls.BACKEND_GET_FILE_DATA + file_id, { params });
    return response.data;
  } catch (error) {
    logger.error(error);
    throw error;
  }
};

/**
 * Applies column transformations to a dataset and creates a new file.
 *
 * @param {string} file_id - ID of the source file.
 * @param {Array} transformations - List of transformation descriptors.
 * @returns {Promise<object>} API response with success status and message.
 */
export const transformData = async (file_id, transformations) => {
  const data = {
    file_id: file_id,
    transformations,
  };

  return axios
    .post(urls.BACKEND_TRANSFORM_DATA + file_id, data)
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data;
      }
      return resp.data;
    })
    .catch((err) => {
      logger.error(err);
      throw err.response?.data || err;
    });
};

/**
 * Fetches per-column descriptive statistics for a file.
 *
 * @param {string} file_id - ID of the uploaded file.
 * @returns {Promise<Array>} Array of stat objects, one per column.
 */
export const getColumnStats = async (file_id) =>
  axios
    .get(urls.BACKEND_GET_COLUMN_STATS + file_id)
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

/**
 * Deletes an uploaded file by ID.
 *
 * @param {string} fileId
 * @returns {Promise<object>} API response.
 */
export const deleteFile = async (fileId) => {
  return axios
    .delete(urls.BACKEND_DELETE_FILE + fileId)
    .then((response) => {
      return response.data;
    })
    .catch((error) => {
      logger.error("Error:", error);
      throw error;
    });
};

