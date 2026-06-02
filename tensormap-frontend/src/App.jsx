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

function withErrorBoundary(Wrapped) {
  const Wrapper = (props) => (
    <ErrorBoundary>
      <Wrapped {...props} />
    </ErrorBoundary>
  );
  Wrapper.displayName = `withErrorBoundary(${Wrapped.displayName || Wrapped.name || "Component"})`;
  return Wrapper;
}

const ProjectsPageWithBoundary = withErrorBoundary(ProjectsPage);
const DataUploadWithBoundary = withErrorBoundary(DataUpload);
const DataProcessWithBoundary = withErrorBoundary(DataProcess);
const DeepLearningWithBoundary = withErrorBoundary(DeepLearning);
const TrainingWithBoundary = withErrorBoundary(Training);

/**
 * Root application component that defines all client-side routes.
 *
 * Uses lazy-loaded page components wrapped in per-route ErrorBoundary components
 * and a Suspense fallback. Each route is isolated — if one page crashes, the
 * others remain functional. An outer ErrorBoundary wraps the entire route tree
 * so that WorkspaceLayout and the router shell also have a safety net.
 */
function App() {
  return (
    <BrowserRouter>
      <AppTopBar />
      <Suspense fallback={<PageLoader />}>
        <ErrorBoundary>
          <Routes>
            {/* Projects landing page */}
            <Route path="/projects" element={<ProjectsPageWithBoundary />} />

            {/* Workspace with sidebar */}
            <Route path="/workspace/:projectId" element={<WorkspaceLayout />}>
              <Route index element={<Navigate to="dataset" replace />} />
              <Route path="dataset" element={<DataUploadWithBoundary />} />
              <Route path="datasets" element={<Navigate to="../dataset" replace />} />
              <Route path="process" element={<DataProcessWithBoundary />} />
              <Route path="models" element={<DeepLearningWithBoundary />} />
              <Route path="training" element={<TrainingWithBoundary />} />
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
        </ErrorBoundary>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
