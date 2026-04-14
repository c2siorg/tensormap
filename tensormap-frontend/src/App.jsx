import { Suspense, lazy } from "react";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import { ErrorBoundary, ErrorFallback, PageLoader } from "./components/ErrorBoundary/ErrorBoundary";
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
 * Uses lazy-loaded page components with Suspense and error boundaries.
 * Routes are organised into a projects listing and a per-project workspace
 * layout with nested dataset, process, model, and training pages.
 */
function App() {
  return (
    <BrowserRouter>
      <AppTopBar />
      <ErrorBoundary>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            {/* Projects landing page */}
            <Route
              path="/projects"
              element={
                <ErrorBoundary>
                  <ProjectsPage />
                </ErrorBoundary>
              }
            />

            {/* Workspace with sidebar */}
            <Route path="/workspace/:projectId" element={<WorkspaceLayout />}>
              <Route index element={<Navigate to="dataset" replace />} />
              <Route
                path="dataset"
                element={
                  <ErrorBoundary>
                    <DataUpload />
                  </ErrorBoundary>
                }
              />
              <Route path="datasets" element={<Navigate to="../dataset" replace />} />
              <Route
                path="process"
                element={
                  <ErrorBoundary>
                    <DataProcess />
                  </ErrorBoundary>
                }
              />
              <Route
                path="models"
                element={
                  <ErrorBoundary>
                    <DeepLearning />
                  </ErrorBoundary>
                }
              />
              <Route
                path="training"
                element={
                  <ErrorBoundary>
                    <Training />
                  </ErrorBoundary>
                }
              />
            </Route>

            {/* Legacy redirects */}
            <Route path="/" element={<Navigate to="/projects" replace />} />
            <Route path="/home" element={<Navigate to="/projects" replace />} />
            <Route path="/data-upload" element={<Navigate to="/projects" replace />} />
            <Route path="/data-process" element={<Navigate to="/projects" replace />} />
            <Route path="/deep-learning" element={<Navigate to="/projects" replace />} />

            {/* Catch-all */}
            <Route
              path="*"
              element={<ErrorFallback error={{ message: "Page not found" }} onGoHome={() => {}} />}
            />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </BrowserRouter>
  );
}

export default App;
