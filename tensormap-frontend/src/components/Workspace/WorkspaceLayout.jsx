import { Outlet } from "react-router-dom";
import WorkspaceSidebar from "./WorkspaceSidebar";
import WorkspaceTopBar from "./WorkspaceTopBar";
import useProjectData from "../../hooks/useProjectData";

export default function WorkspaceLayout() {
  const { project } = useProjectData();

  return (
    <div className="flex h-screen">
      <WorkspaceSidebar projectName={project?.name} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <WorkspaceTopBar projectName={project?.name} />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
