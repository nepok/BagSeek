import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ErrorProvider } from './components/ErrorContext/ErrorContext';
import { ThemeProvider } from '@mui/material/styles';
import darkTheme from './theme';
import { BrowserRouter } from 'react-router-dom'; // ⬅️ neu

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <BrowserRouter /* basename optional, siehe Hinweis unten */>
      <ThemeProvider theme={darkTheme}>
        <ErrorProvider>
          <App />
        </ErrorProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);

reportWebVitals();