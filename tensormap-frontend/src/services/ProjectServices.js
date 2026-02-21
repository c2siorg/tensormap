import axios from "../shared/Axios";
import * as urls from "../constants/Urls";

/**
 * Fetches all projects.
 *
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export const getAllProjects = () => axios.get(urls.BACKEND_PROJECT);

/**
 * Fetches a single project by ID.
 *
 * @param {string | number} id
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export const getProject = (id) => axios.get(`${urls.BACKEND_PROJECT}/${id}`);

/**
 * Creates a new project.
 *
 * @param {{ name: string, description?: string | null }} data
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export const createProject = (data) => axios.post(urls.BACKEND_PROJECT, data);
