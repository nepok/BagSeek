import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { Box, CircularProgress } from '@mui/material';

const PrivateRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading, authDisabled } = useAuth();

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', bgcolor: '#121212' }}>
        <CircularProgress sx={{ color: '#B49FCC' }} />
      </Box>
    );
  }

  if (authDisabled || isAuthenticated) return <>{children}</>;

  return <Navigate to="/login" replace />;
};

export default PrivateRoute;
