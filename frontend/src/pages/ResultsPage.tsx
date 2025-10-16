import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysisStore } from '../store/analysisStore';
import ResultsDisplay from '../components/ResultsDisplay';

const ResultsPage: React.FC = () => {
  const { results } = useAnalysisStore();
  const navigate = useNavigate();

  if (!results) {
    navigate('/analysis');
    return null;
  }

  const handleRetakePhoto = () => {
    navigate('/analysis');
  };

  const handleBackToWelcome = () => {
    navigate('/');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis Results</h1>
        <p className="text-gray-600">Review your skin analysis and recommendations</p>
      </div>

      <ResultsDisplay
        results={results}
        originalImage={results.originalImage}
        onRetakePhoto={handleRetakePhoto}
        onBackToWelcome={handleBackToWelcome}
      />
    </div>
  );
};

export default ResultsPage;
