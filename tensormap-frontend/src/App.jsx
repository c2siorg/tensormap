import { Suspense, lazy } from "react";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import AppTopBar from "./components/AppTopBar";
import ErrorBoundary, { PageLoader } from "./components/ErrorBoundary";
import WorkspaceLayout from "./components/Workspace/WorkspaceLayout";

const ProjectsPage = lazy(() => import("./containers/ProjectsPage/ProjectsPage"));
const DataUpload = lazy(() => import("./containers/DataUpload/DataUpload"));
const DataProcess = lazy(() => import("./containers/DataProcess/DataProcess"));
const DeepLearning = lazy(() => import("./containers/DeepLearning/DeepLearning"));
const Training = lazy(() => import("./containers/Training/Training"));

const withErrorBoundary = (Component) => (
  <ErrorBoundary>
    <Component />
  </ErrorBoundary>
);

/**
 * Root application component that defines all client-side routes.
 *
 * Uses lazy-loaded page components wrapped in per-route ErrorBoundary components
 * and a Suspense fallback. Each route is isolated — if one page crashes, the
 * others remain functional.
 */
function App() {
  return (
    <BrowserRouter>
      <AppTopBar />
      <Suspense fallback={<PageLoader />}>
        <Routes>
          {/* Projects landing page */}
          <Route path="/projects" element={withErrorBoundary(ProjectsPage)} />

          {/* Workspace with sidebar */}
          <Route path="/workspace/:projectId" element={<WorkspaceLayout />}>
            <Route index element={<Navigate to="dataset" replace />} />
            <Route path="dataset" element={withErrorBoundary(DataUpload)} />
            <Route path="datasets" element={<Navigate to="../dataset" replace />} />
            <Route path="process" element={withErrorBoundary(DataProcess)} />
            <Route path="models" element={withErrorBoundary(DeepLearning)} />
            <Route path="training" element={withErrorBoundary(Training)} />
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
