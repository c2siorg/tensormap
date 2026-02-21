import { useEffect } from "react";
import { useParams } from "react-router-dom";
import { useRecoilState } from "recoil";
import { currentProject, projectFiles } from "../shared/atoms";
import { getProject } from "../services/ProjectServices";
import { getAllFiles } from "../services/FileServices";
import logger from "../shared/logger";

/**
 * Loads project metadata and file list for the current workspace.
 *
 * Reads `:projectId` from the URL, fetches the project and its files
 * from the API, and stores them in Recoil atoms. Returns the current
 * project, files, and projectId so workspace components can access
 * them without duplicating fetch logic.
 *
 * @returns {{ project: object | null, files: Array, projectId: string }}
 */
export default function useProjectData() {
  const { projectId } = useParams();
  const [project, setProject] = useRecoilState(currentProject);
  const [files, setFiles] = useRecoilState(projectFiles);

  useEffect(() => {
    if (!projectId) return;

    getProject(projectId)
      .then((resp) => {
        if (resp.data.success) {
          setProject(resp.data.data);
        }
      })
      .catch((err) => logger.error("Failed to load project:", err));

    getAllFiles(projectId)
      .then((response) => {
        setFiles(response);
      })
      .catch((err) => logger.error("Failed to load project files:", err));
  }, [projectId, setProject, setFiles]);

  return { project, files, projectId };
}
