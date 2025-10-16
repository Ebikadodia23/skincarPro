import React from 'react';
import { Link } from 'react-router-dom';

const DashboardPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
        <p className="text-gray-600">Welcome back! Ready for your next skin analysis?</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Link
          to="/analysis"
          className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow p-6 border border-gray-200"
        >
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">New Analysis</h3>
          <p className="text-gray-600 text-sm">Start a new skin analysis session</p>
        </Link>

        <Link
          to="/history"
          className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow p-6 border border-gray-200"
        >
          <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Analysis History</h3>
          <p className="text-gray-600 text-sm">View your past skin analyses</p>
        </Link>

        <Link
          to="/profile"
          className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow p-6 border border-gray-200"
        >
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Profile Settings</h3>
          <p className="text-gray-600 text-sm">Manage your account and preferences</p>
        </Link>
      </div>
    </div>
  );
};

export default DashboardPage;
