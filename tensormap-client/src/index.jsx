import ReactDOM from "react-dom";
import "./index.css";
import "semantic-ui-css/semantic.min.css";
import { RecoilRoot } from "recoil";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
ReactDOM.createRoot(document.getElementById("root")).render(
  <RecoilRoot>
    <App />
  </RecoilRoot>,
);

reportWebVitals();
