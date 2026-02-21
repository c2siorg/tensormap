import { useEffect } from "react";
import { useParams } from "react-router-dom";
import { useRecoilState } from "recoil";
import { currentProject, projectFiles } from "../shared/atoms";
import { getProject } from "../services/ProjectServices";
import { getAllFiles } from "../services/FileServices";

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
      .catch((err) => console.error("Failed to load project:", err));

    getAllFiles(projectId)
      .then((response) => {
        setFiles(response);
      })
      .catch((err) => console.error("Failed to load project files:", err));
  }, [projectId, setProject, setFiles]);

  return { project, files, projectId };
}
