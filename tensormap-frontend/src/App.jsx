import { Suspense, lazy } from "react";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import AppTopBar from "./components/AppTopBar";
import WorkspaceLayout from "./components/Workspace/WorkspaceLayout";

const ProjectsPage = lazy(() => import("./containers/ProjectsPage/ProjectsPage"));
const DataUpload = lazy(() => import("./containers/DataUpload/DataUpload"));
const DataProcess = lazy(() => import("./containers/DataProcess/DataProcess"));
const DeepLearning = lazy(() => import("./containers/DeepLearning/DeepLearning"));
const Training = lazy(() => import("./containers/Training/Training"));

/**
 * Root application component that defines all client-side routes.
 *
 * Uses lazy-loaded page components with a Suspense fallback. Routes are
 * organised into a projects listing and a per-project workspace layout
 * with nested dataset, process, model, and training pages.
 */
function App() {
  return (
    <BrowserRouter>
      <AppTopBar />
      <Suspense
        fallback={<div className="flex h-screen items-center justify-center">Loading...</div>}
      >
        <Routes>
          {/* Projects landing page */}
          <Route path="/projects" element={<ProjectsPage />} />

          {/* Workspace with sidebar */}
          <Route path="/workspace/:projectId" element={<WorkspaceLayout />}>
            <Route index element={<Navigate to="dataset" replace />} />
            <Route path="dataset" element={<DataUpload />} />
            <Route path="datasets" element={<Navigate to="../dataset" replace />} />
            <Route path="process" element={<DataProcess />} />
            <Route path="models" element={<DeepLearning />} />
            <Route path="training" element={<Training />} />
          </Route>

          {/* Legacy redirects */}
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/home" element={<Navigate to="/projects" replace />} />
          <Route path="/data-upload" element={<Navigate to="/projects" replace />} />
          <Route path="/data-process" element={<Navigate to="/projects" replace />} />
          <Route path="/deep-learning" element={<Navigate to="/projects" replace />} />

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
