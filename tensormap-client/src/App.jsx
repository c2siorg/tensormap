import { Suspense } from "react";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import Home from "./containers/Home/Home";
import DataUpload from "./containers/DataUpload/DataUpload";
import DataProcess from "./containers/DataProcess/DataProcess";
import DeepLearning from "./containers/DeepLearning/DeepLearning";
import * as urls from "./constants/Urls";
import Layout from "./components/Layout/Layout";
import NavBar from "./components/Layout/NavBar/NavBar";

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route
            exact
            path={urls.HOME_URL}
            element={
              <Home />
            }
          />
          <Route
            path={urls.DATA_UPLOAD_URL}
            element={
              <DataUpload />
            }
          />
          <Route
            path={urls.DATA_PROCESS_URL}
            element={
              <DataProcess />
            }
          />
          <Route
            path={urls.DEEP_LEARN_URL}
            element={
              <DeepLearning />
            }
          />
          <Route exact path="/" element={<Navigate to={urls.HOME_URL} />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
