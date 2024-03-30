import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import 'semantic-ui-css/semantic.min.css';
import { RecoilRoot } from 'recoil';
import App from './App';
import reportWebVitals from './reportWebVitals';
createRoot(document.getElementById('root')).render(
  <RecoilRoot>
    <App />
  </RecoilRoot>,
)

reportWebVitals();
