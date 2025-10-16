import React from 'react';
import { Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

const HomePage: React.FC = () => {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 to-purple-700">
      <div className="max-w-6xl mx-auto px-4 py-16">
        <div className="text-center text-white mb-12">
          <div className="mb-8">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-500/20 backdrop-blur-sm text-blue-100 text-sm font-medium mb-6">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
              </svg>
              Next-Generation Skin Analysis
            </div>
            <h1 className="text-5xl font-bold mb-6">
              Understand Your <span className="bg-gradient-to-r from-blue-200 to-purple-200 bg-clip-text text-transparent">Skin</span>
            </h1>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Get personalized skin analysis with AI-powered precision. Detect conditions, understand patterns, and receive expert recommendations.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 text-white">
              <div className="w-12 h-12 bg-blue-500/30 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2">Smart Capture</h3>
              <p className="text-blue-100 text-sm">Advanced guidance for optimal image quality with real-time feedback</p>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 text-white">
              <div className="w-12 h-12 bg-purple-500/30 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2">AI Analysis</h3>
              <p className="text-blue-100 text-sm">Detect 6+ skin conditions with explainable AI and confidence scoring</p>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 text-white">
              <div className="w-12 h-12 bg-green-500/30 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2">Expert Guidance</h3>
              <p className="text-blue-100 text-sm">Personalized recommendations and optional clinician review</p>
            </div>
          </div>

          <Link
            to={isAuthenticated ? "/analysis" : "/register"}
            className="inline-flex items-center px-8 py-4 rounded-full bg-white text-blue-600 font-semibold text-lg shadow-lg hover:shadow-xl transition-all transform hover:scale-105"
          >
            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
            </svg>
            Start Skin Analysis
          </Link>

          <p className="text-sm text-blue-200 mt-6 max-w-2xl mx-auto">
            Your privacy is protected. Images are processed securely and you control data usage.
            This tool provides cosmetic recommendations and is not a medical diagnostic device.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
