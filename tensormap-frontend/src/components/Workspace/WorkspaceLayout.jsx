import { Outlet } from "react-router-dom";
import WorkspaceSidebar from "./WorkspaceSidebar";
import WorkspaceTopBar from "./WorkspaceTopBar";
import useProjectData from "../../hooks/useProjectData";

/**
 * Shell layout for a single project workspace.
 *
 * Renders the sidebar navigation, a breadcrumb top bar, and an Outlet
 * for the active nested route (datasets, process, models, training).
 */
export default function WorkspaceLayout() {
  const { project } = useProjectData();

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
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
