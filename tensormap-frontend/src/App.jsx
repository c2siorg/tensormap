import { Suspense } from "react";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import ProjectsPage from "./containers/ProjectsPage/ProjectsPage";
import WorkspaceLayout from "./components/Workspace/WorkspaceLayout";
import DataUpload from "./containers/DataUpload/DataUpload";
import DataProcess from "./containers/DataProcess/DataProcess";
import DeepLearning from "./containers/DeepLearning/DeepLearning";
import Training from "./containers/Training/Training";

function App() {
  return (
    <BrowserRouter>
      <Suspense
        fallback={<div className="flex h-screen items-center justify-center">Loading...</div>}
      >
        <Routes>
          {/* Projects landing page */}
          <Route path="/projects" element={<ProjectsPage />} />

          {/* Workspace with sidebar */}
          <Route path="/workspace/:projectId" element={<WorkspaceLayout />}>
            <Route index element={<Navigate to="datasets" replace />} />
            <Route path="datasets" element={<DataUpload />} />
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
