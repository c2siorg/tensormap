import axios from "../shared/Axios";
import * as urls from "../constants/Urls";

export const getAllFiles = async () =>
  axios
    .get(urls.BACKEND_GET_ALL_FILES)
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.data;
      }
      return [];
    })
    .catch((err) => {
      console.error(err);
      throw err;
    });

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append("data", file);

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
      console.error(err);
      throw err;
    });
};

export const setTargetField = async (fileId, targetField) => {
  const data = {
    file_id: fileId,
    target: targetField,
  };

  return axios
    .post(urls.BACKEND_ADD_TARGET_FIELD, data)
    .then((resp) => {
      if (resp.data.success === true) {
        return true;
      }
      return false;
    })
    .catch((err) => {
      console.error(err.response.data.message);
      return false;
    });
};

export const getCovMatrix = async (file_id) =>
  axios
    .get(urls.BACKEND_GET_COV_MATRIX + file_id)
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.data;
      }
    })
    .catch((err) => {
      console.error(err);
      throw err;
    });

export const getFileData = async (file_id) => {
  try {
    console.log(urls.BACKEND_GET_FILE_DATA + file_id);
    const response = await axios.get(urls.BACKEND_GET_FILE_DATA + file_id);
    return response.data.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const transformData = async (file_id, transformations) => {
  const data = {
    file_id: file_id,
    transformations,
  };

  return axios
    .post(urls.BACKEND_TRANSFORM_DATA + file_id, data)
    .then((resp) => {
      console.log(resp);
      if (resp.data.success === true) {
        return resp.data;
      }
      return resp.data;
    })
    .catch((err) => {
      console.error(err);
      throw err.data;
    });
};

export const deleteFile = async (fileId) => {
  console.log(fileId);
  axios
    .delete(urls.BACKEND_DELETE_FILE + fileId)
    .then((response) => {
      console.log(response.data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
};

export const submitImageParameters = async (data) => {
  console.log(data.fileId)
    axios
      .post(urls.BACKEND_IMAGE_PREPROCESS + data.fileId, data)
      .then((response) => {
        console.log("Success:", response.data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
};