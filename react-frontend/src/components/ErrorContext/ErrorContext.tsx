import React, { createContext, useState, useContext, ReactNode } from 'react';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';

interface ErrorContextType {
  setError: (message: string) => void;
}

// Create a context for error handling, initially undefined
const ErrorContext = createContext<ErrorContextType | undefined>(undefined);

// Custom hook to access the error context, throws if used outside provider
export const useError = (): ErrorContextType => {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useError must be used within an ErrorProvider');
  }
  return context;
};

// Provider component that manages error state and displays error messages
export const ErrorProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // State to hold the current error message or null if no error
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Handler to clear the error message, closes Snackbar and Alert
  const handleClose = () => {
    setErrorMessage(null);
  };

  return (
    // Provide the setError function to children components
    <ErrorContext.Provider value={{ setError: setErrorMessage }}>
      {children}
      {/* Snackbar shows up when errorMessage is set, auto hides after 6 seconds */}
      <Snackbar
        open={!!errorMessage}
        autoHideDuration={6000}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        {/* Alert component displays the error message with severity "error" */}
        <Alert onClose={handleClose} severity="error" sx={{ width: '100%' }}>
          {errorMessage}
        </Alert>
      </Snackbar>
    </ErrorContext.Provider>
  );
};