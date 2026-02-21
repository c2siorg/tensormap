import axios from "../shared/Axios";

export const getAllProjects = () => axios.get("/project");

export const getProject = (id) => axios.get(`/project/${id}`);

export const createProject = (data) => axios.post("/project", data);

export const updateProject = (id, data) => axios.patch(`/project/${id}`, data);

export const deleteProject = (id) => axios.delete(`/project/${id}`);
