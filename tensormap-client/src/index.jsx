import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import 'semantic-ui-css/semantic.min.css';
import { RecoilRoot } from 'recoil';
import App from './App';
import reportWebVitals from './reportWebVitals';

ReactDOM.render(
  <StrictMode>
  <RecoilRoot>
    <App />
  </RecoilRoot></StrictMode>,
  document.getElementById('root'),
);

reportWebVitals();
