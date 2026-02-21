export const base_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:4300/api/v1";
export const BACKEND_GET_ALL_FILES = "/data/upload/file";
export const BACKEND_GET_ALL_MODELS = "/model/model-list";
export const BACKEND_DOWNLOAD_CODE = "/model/code";
export const BACKEND_RUN_MODEL = "/model/run";
export const BACKEND_ADD_TARGET_FIELD = "/data/process/target";
export const BACKEND_FILE_UPLOAD = "/data/upload/file";
export const WS_DL_RESULTS = import.meta.env.VITE_WS_URL || "http://127.0.0.1:4300/dl-result";
export const BACKEND_VALIDATE_MODEL = "/model/validate";
export const BACKEND_GET_COV_MATRIX = "/data/process/data_metrics/";
export const BACKEND_GET_FILE_DATA = "/data/process/file/";
export const BACKEND_TRANSFORM_DATA = "/data/process/preprocess/";
export const BACKEND_DELETE_FILE = "/data/upload/file/";
export const BACKEND_IMAGE_PREPROCESS = "/data/process/image_preprocess/";

// Project routes
export const PROJECTS_URL = "/projects";
export const projectWorkspaceUrl = (projectId) => `/projects/${projectId}`;

// Project backend endpoints
export const BACKEND_PROJECTS = "/project";
export const BACKEND_PROJECT_BY_ID = (id) => `/project/${id}`;
