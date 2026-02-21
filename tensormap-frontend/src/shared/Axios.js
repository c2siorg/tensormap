import axios from "axios";
import * as urls from "../constants/Urls";
import logger from "./logger";

/**
 * Pre-configured Axios instance for all backend API calls.
 *
 * Base URL is read from the `VITE_API_BASE_URL` env var (defaults to
 * `http://127.0.0.1:4300/api/v1`). Includes request and response
 * interceptors that log outgoing calls and errors via the logger.
 */
const Axios = axios.create({ baseURL: urls.base_URL });

Axios.interceptors.request.use((config) => {
  logger.debug(`${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
  return config;
});

Axios.interceptors.response.use(
  (response) => response,
  (error) => {
    logger.error(
      `Request failed: ${error.config?.method?.toUpperCase()} ${error.config?.url}`,
      error.response?.status ?? "no status",
      error.message,
    );
    return Promise.reject(error);
  },
);

export default Axios;
