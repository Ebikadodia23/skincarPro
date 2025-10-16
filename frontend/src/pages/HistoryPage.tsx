import React from 'react';

const HistoryPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis History</h1>
        <p className="text-gray-600">View your past skin analyses</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm p-8 border border-gray-200 text-center">
        <svg className="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analysis History Yet</h3>
        <p className="text-gray-600 mb-4">Start your first analysis to see results here</p>
      </div>
    </div>
  );
};

export default HistoryPage;
