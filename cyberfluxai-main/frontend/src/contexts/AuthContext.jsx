import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext(null);

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('cyberflux_token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchCurrentUser();
    } else {
      setLoading(false);
    }
  }, [token]);

  const fetchCurrentUser = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      setUser(response.data);
    } catch (error) {
      console.error('Failed to fetch user:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (username, password) => {
    try {
      console.log('AuthContext: Logging in user:', username);
      
      const response = await axios.post(`${BACKEND_URL}/api/auth/login`, {
        username,
        password
      });
      
      console.log('AuthContext: Login response received');
      
      const { access_token } = response.data;
      
      if (!access_token) {
        throw new Error('No access token received');
      }
      
      console.log('AuthContext: Token received, saving...');
      setToken(access_token);
      localStorage.setItem('cyberflux_token', access_token);
      
      console.log('AuthContext: Fetching user data...');
      // Fetch user data
      const userResponse = await axios.get(`${BACKEND_URL}/api/auth/me`, {
        headers: {
          Authorization: `Bearer ${access_token}`
        }
      });
      
      console.log('AuthContext: User data received:', userResponse.data);
      setUser(userResponse.data);
      
      return userResponse.data;
    } catch (error) {
      console.error('AuthContext: Login error:', error);
      throw error;
    }
  };

  const register = async (email, username, fullName, password) => {
    const response = await axios.post(`${BACKEND_URL}/api/auth/register`, {
      email,
      username,
      full_name: fullName,
      password
    });
    return response.data;
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('cyberflux_token');
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};