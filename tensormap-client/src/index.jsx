import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import 'semantic-ui-css/semantic.min.css';
import { RecoilRoot } from 'recoil';
import App from './App';
import reportWebVitals from './reportWebVitals';

ReactDOM.render(
 
  <RecoilRoot>
    <App />
  </RecoilRoot>,
  document.getElementById('root'),
);

reportWebVitals();
