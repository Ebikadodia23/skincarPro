import React from 'react';
import { useNavigate } from 'react-router-dom';
import CameraCapture from '../components/CameraCapture';
import { useAnalysisStore } from '../store/analysisStore';

const AnalysisPage: React.FC = () => {
  const navigate = useNavigate();
  const { setResults } = useAnalysisStore();

  const handleImageCapture = async (blob: Blob, dataUrl: string) => {
    try {
      const formData = new FormData();
      formData.append('file', blob);

      const uploadResponse = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const uploadResult = await uploadResponse.json();

      const predictionResponse = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: uploadResult.file_id })
      });

      if (!predictionResponse.ok) {
        throw new Error('Analysis failed');
      }

      const results = await predictionResponse.json();
      setResults({ ...results, originalImage: dataUrl });
      navigate('/results');
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Analysis failed. Please try again.');
    }
  };

  const handleBack = () => {
    navigate('/dashboard');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Skin Analysis</h1>
        <p className="text-gray-600">Capture or upload an image for AI-powered analysis</p>
      </div>

      <CameraCapture onImageCapture={handleImageCapture} onBack={handleBack} />
    </div>
  );
};

export default AnalysisPage;
