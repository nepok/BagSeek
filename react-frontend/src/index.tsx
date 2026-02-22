import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ErrorProvider } from './components/ErrorContext/ErrorContext';
import { ThemeProvider } from '@mui/material/styles';
import darkTheme from './theme';
import { BrowserRouter } from 'react-router-dom';
// import { AuthProvider } from './components/AuthContext/AuthContext';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider theme={darkTheme}>
        {/* <AuthProvider> */}
          <ErrorProvider>
            <App />
          </ErrorProvider>
        {/* </AuthProvider> */}
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);

reportWebVitals();