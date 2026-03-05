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

/**
 * Partially updates a project's name and/or description.
 *
 * @param {string} id
 * @param {{ name?: string, description?: string | null }} data
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export const updateProject = (id, data) => axios.patch(`${urls.BACKEND_PROJECT}/${id}`, data);

/**
 * Deletes a project and all its associated files and models.
 *
 * @param {string} id
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export const deleteProject = (id) => axios.delete(`${urls.BACKEND_PROJECT}/${id}`);
