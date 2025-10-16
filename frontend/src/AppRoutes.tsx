import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// Components
import Header from './components/Header';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import AnalysisPage from './pages/AnalysisPage';
import ResultsPage from './pages/ResultsPage';
import ProfilePage from './pages/ProfilePage';
import HistoryPage from './pages/HistoryPage';
import LoadingSpinner from './components/LoadingSpinner';

// Store
import { useAuthStore } from './store/authStore';
import { useAnalysisStore } from './store/analysisStore';

// Types
export interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  age?: number;
  skin_type?: string;
  skin_concerns?: string[];
  fitzpatrick_type?: number;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
}

export interface PredictionResults {
  id?: string;
  predictions: Record<string, number>;
  landmarks?: Array<{
    id: number;
    x: number;
    y: number;
    z?: number;
    zone: string;
  }>;
  heatmap_data?: string;
  heatmap_b64?: string;
  explanations?: Array<{
    condition: string;
    score: number;
    severity: string;
    explanation: string;
    recommendations: string[];
  }>;
  confidence_score?: number;
  confidence?: string;
  timestamp?: string;
  zone_scores?: Record<string, number>;
  skin_tone_analysis?: Record<string, any>;
  dominant_conditions?: string[];
  severity_assessment?: Record<string, string>;
  recommendations?: Array<{
    product_id: string;
    condition: string;
    severity_score: number;
    recommendation_strength: number;
    reason: string;
  }>;
  processing_time?: number;
  created_at?: string;
  originalImage?: string;
}

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuthStore();
  
  if (isLoading) {
    return <LoadingSpinner message="Checking authentication..." />;
  }
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuthStore();
  
  if (isLoading) {
    return <LoadingSpinner message="Loading..." />;
  }
  
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }
  
  return <>{children}</>;
};

const AppRoutes: React.FC = () => {
  const location = useLocation();
  const { initializeAuth } = useAuthStore();
  const { clearResults } = useAnalysisStore();
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const initialize = async () => {
      await initializeAuth();
      setIsInitialized(true);
    };
    
    initialize();
  }, [initializeAuth]);

  useEffect(() => {
    // Clear analysis results when navigating away from results page
    if (location.pathname !== '/results') {
      clearResults();
    }
  }, [location.pathname, clearResults]);

  if (!isInitialized) {
    return <LoadingSpinner message="Initializing app..." />;
  }

  const hideHeaderPaths = ['/login', '/register'];
  const showHeader = !hideHeaderPaths.includes(location.pathname);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {showHeader && <Header />}
      
      <main className={showHeader ? "pt-16" : ""}>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={
            <PublicRoute>
              <LoginPage />
            </PublicRoute>
          } />
          <Route path="/register" element={
            <PublicRoute>
              <RegisterPage />
            </PublicRoute>
          } />
          
          {/* Protected routes */}
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          } />
          <Route path="/analysis" element={
            <ProtectedRoute>
              <AnalysisPage />
            </ProtectedRoute>
          } />
          <Route path="/results" element={
            <ProtectedRoute>
              <ResultsPage />
            </ProtectedRoute>
          } />
          <Route path="/profile" element={
            <ProtectedRoute>
              <ProfilePage />
            </ProtectedRoute>
          } />
          <Route path="/history" element={
            <ProtectedRoute>
              <HistoryPage />
            </ProtectedRoute>
          } />
          
          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
            borderRadius: '12px',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
          },
          success: {
            style: {
              background: '#10b981',
            },
          },
          error: {
            style: {
              background: '#ef4444',
            },
          },
        }}
      />
    </div>
  );
};

export default AppRoutes;