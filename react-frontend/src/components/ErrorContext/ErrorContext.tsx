import React, { createContext, useState, useContext, useCallback, ReactNode } from 'react';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

type NotificationSeverity = 'error' | 'info' | 'success' | 'warning';

interface Notification {
  message: string;
  severity: NotificationSeverity;
  loading?: boolean;
}

interface ErrorContextType {
  setError: (message: string) => void;
  setInfo: (message: string, loading?: boolean) => void;
  clearNotification: () => void;
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
  // State to hold the current notification or null if none
  const [notification, setNotification] = useState<Notification | null>(null);

  // Handler to clear the notification
  const handleClose = useCallback(() => {
    setNotification(null);
  }, []);

  // Set an error message
  const setError = useCallback((message: string) => {
    setNotification({ message, severity: 'error' });
  }, []);

  // Set an info message (optionally with loading indicator)
  const setInfo = useCallback((message: string, loading?: boolean) => {
    setNotification({ message, severity: 'info', loading });
  }, []);

  return (
    // Provide the notification functions to children components
    <ErrorContext.Provider value={{ setError, setInfo, clearNotification: handleClose }}>
      {children}
      {/* Snackbar shows up when notification is set */}
      <Snackbar
        open={!!notification}
        autoHideDuration={notification?.loading ? null : 6000}
        onClose={notification?.loading ? undefined : handleClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={notification?.loading ? undefined : handleClose}
          severity={notification?.severity || 'error'}
          sx={{ width: '100%' }}
          icon={notification?.loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <CircularProgress size={20} color="inherit" />
            </Box>
          ) : undefined}
        >
          {notification?.message}
        </Alert>
      </Snackbar>
    </ErrorContext.Provider>
  );
};