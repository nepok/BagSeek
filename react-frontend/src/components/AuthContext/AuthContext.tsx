import React, { createContext, useState, useContext, useEffect, useCallback, ReactNode } from 'react';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  authDisabled: boolean;
  login: (password: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [authDisabled, setAuthDisabled] = useState(false);

  const checkAuth = useCallback(async () => {
    try {
      const res = await fetch('/api/auth-check', { credentials: 'include' });
      const data = await res.json();
      setIsAuthenticated(data.authenticated === true);
      setAuthDisabled(data.authDisabled === true);
    } catch {
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { checkAuth(); }, [checkAuth]);

  // Intercept 401 responses from any fetch call to trigger re-login
  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = async (...args: Parameters<typeof fetch>): Promise<Response> => {
      const response = await originalFetch(...args);
      if (response.status === 401) {
        const url = typeof args[0] === 'string' ? args[0] : (args[0] as Request).url;
        if (!url.includes('/api/auth-check') && !url.includes('/api/login')) {
          setIsAuthenticated(false);
        }
      }
      return response;
    };
    return () => { window.fetch = originalFetch; };
  }, []);

  const login = useCallback(async (password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (res.ok) {
        setIsAuthenticated(true);
        return { success: true };
      }
      return { success: false, error: data.error || 'Login failed' };
    } catch {
      return { success: false, error: 'Network error' };
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await fetch('/api/logout', { method: 'POST', credentials: 'include' });
    } catch {
      // Still logout client-side even if server call fails
    }
    setIsAuthenticated(false);
  }, []);

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, authDisabled, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
