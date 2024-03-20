import { Suspense } from "react";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import Home from "./containers/Home/Home";
import DataUpload from "./containers/DataUpload/DataUpload";
import DataProcess from "./containers/DataProcess/DataProcess";
import DeepLearning from "./containers/DeepLearning/DeepLearning";
import * as urls from "./constants/Urls";
import SideBar from "./components/Layout/SideBar/SideBar";
import Layout from "./components/Layout/Layout";

function App() {
    return (
        <BrowserRouter>
            <Suspense fallback={<div>Loading...</div>}>
                <Routes>
                        <Route exact path={urls.HOME_URL} element={<Layout > <Home /></Layout>} />
                        <Route path={urls.DATA_UPLOAD_URL} element={<Layout><DataUpload /></Layout>} />
                        <Route path={urls.DATA_PROCESS_URL} element={<Layout><DataProcess/></Layout>} />
                        <Route path={urls.DEEP_LEARN_URL} element={<Layout><DeepLearning /></Layout>}/>
                        <Route exact path="/" element={<Navigate to={urls.HOME_URL} />} />
                </Routes>
            </Suspense>
        </BrowserRouter>
    );
}

export default App;
