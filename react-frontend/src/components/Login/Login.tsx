import React, { useState, FormEvent } from 'react';
import { Navigate } from 'react-router-dom';
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  Paper,
  InputAdornment,
  IconButton,
} from '@mui/material';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import { useAuth } from '../AuthContext/AuthContext';
import LoginTractorLoader from '../TractorLoader/LoginTractorLoader';

const Login: React.FC = () => {
  const { isAuthenticated, isLoading, authDisabled, login } = useAuth();
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  if (!isLoading && (isAuthenticated || authDisabled)) {
    return <Navigate to="/explore" replace />;
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);
    const result = await login(password);
    if (!result.success) {
      setError(result.error || 'Invalid password');
      setPassword('');
    }
    setIsSubmitting(false);
  };

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', bgcolor: '#121212' }}>
      <Paper elevation={6} sx={{ p: 4, width: '50%', bgcolor: '#1e1e1e', borderRadius: 2 }}>
        <Typography
          variant="h4"
          sx={{
            mb: 1,
            textAlign: 'center',
            color: '#B49FCC',
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: '2.5rem',
            textTransform: 'uppercase',
          }}
        >
          BagSeek
        </Typography>
        <Typography variant="body2" sx={{ mb: 3, textAlign: 'center', color: 'rgba(255,255,255,0.5)' }}>
          Enter password to continue
        </Typography>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            type={showPassword ? 'text' : 'password'}
            label="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoFocus
            disabled={isSubmitting}
            sx={{ mb: 2 }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowPassword(!showPassword)}
                    edge="end"
                    size="small"
                    sx={{ color: 'rgba(255,255,255,0.5)' }}
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          <Button
            fullWidth
            type="submit"
            variant="contained"
            disabled={isSubmitting || !password}
            sx={{ bgcolor: '#B49FCC', '&:hover': { bgcolor: '#9b84b5' }, textTransform: 'none', fontWeight: 600 }}
          >
            {isSubmitting ? 'Logging in...' : 'Log in'}
          </Button>
          <LoginTractorLoader/>
        </form>
      </Paper>
    </Box>
  );
};

export default Login;
