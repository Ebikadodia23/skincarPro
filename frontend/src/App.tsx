import React, { useState, useRef, useCallback } from 'react';
import { Camera, Upload, Activity, Zap, Eye, Users, BarChart3, Settings } from 'lucide-react';
import CameraCapture from './components/CameraCapture';
import ResultsDisplay from './components/ResultsDisplay';
import ConsentModal from './components/ConsentModal';
import AdminDashboard from './components/AdminDashboard';
import LoadingSpinner from './components/LoadingSpinner';

export interface PredictionResults {
  predictions: Record<string, number>;
  landmarks: Array<{
    id: number;
    x: number;
    y: number;
    z: number;
    zone: string;
  }>;
  heatmap_b64: string;
  explanations: Array<{
    condition: string;
    score: number;
    severity: string;
    explanation: string;
    recommendations: string[];
  }>;
  confidence: string;
  zone_scores: Record<string, number>;
  timestamp: string;
}

const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<'welcome' | 'capture' | 'results' | 'admin'>('welcome');
  const [results, setResults] = useState<PredictionResults | null>(null);
  const [showConsent, setShowConsent] = useState(false);
  const [hasConsent, setHasConsent] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const handleStartAnalysis = () => {
    if (!hasConsent) {
      setShowConsent(true);
    } else {
      setCurrentStep('capture');
    }
  };

  const handleConsentAccept = () => {
    setHasConsent(true);
    setShowConsent(false);
    setCurrentStep('capture');
  };

  const handleImageCapture = async (imageBlob: Blob, imageDataUrl: string) => {
    setIsLoading(true);
    setUploadedImage(imageDataUrl);
    
    try {
      // Upload image
      const formData = new FormData();
      formData.append('file', imageBlob, 'capture.jpg');
      
      const uploadResponse = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        if (errorData.error === 'Image too blurry') {
          throw new Error('Image is too blurry. Please capture a clearer image.');
        }
        throw new Error('Upload failed');
      }
      
      const uploadResult = await uploadResponse.json();
      
      // Run prediction
      const predictionResponse = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ file_id: uploadResult.file_id }),
      });
      
      if (!predictionResponse.ok) {
        throw new Error('Analysis failed');
      }
      
      const predictionResult = await predictionResponse.json();
      setResults(predictionResult);
      setCurrentStep('results');
      
    } catch (error) {
      console.error('Analysis error:', error);
      alert(error instanceof Error ? error.message : 'Analysis failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetakePhoto = () => {
    setResults(null);
    setUploadedImage(null);
    setCurrentStep('capture');
  };

  const handleBackToWelcome = () => {
    setResults(null);
    setUploadedImage(null);
    setCurrentStep('welcome');
  };

  if (isLoading) {
    return <LoadingSpinner message="Analyzing your skin..." />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-lg shadow-sm border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <Activity className="h-5 w-5 text-white" />
                </div>
              </div>
              <div className="ml-3">
                <h1 className="text-xl font-semibold text-gray-900">SkinScan Pro</h1>
                <p className="text-sm text-gray-500">AI-Powered Skin Analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {currentStep !== 'admin' && (
                <button
                  onClick={() => setCurrentStep('admin')}
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
                >
                  <Settings className="h-4 w-4 mr-1" />
                  Admin
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {currentStep === 'welcome' && (
          <div className="text-center">
            <div className="max-w-3xl mx-auto px-4">
              {/* Hero Section */}
              <div className="mb-12">
                <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-100 text-blue-800 text-sm font-medium mb-6">
                  <Zap className="h-4 w-4 mr-2" />
                  Next-Generation Skin Analysis
                </div>
                <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl md:text-6xl mb-6">
                  Understand Your
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"> Skin</span>
                </h1>
                <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                  Get personalized skin analysis with AI-powered precision. Detect conditions, 
                  understand patterns, and receive expert recommendations.
                </p>
              </div>

              {/* Features */}
              <div className="grid md:grid-cols-3 gap-8 mb-12">
                <div className="bg-white/60 backdrop-blur-lg rounded-2xl p-6 border border-gray-100 shadow-sm">
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4 mx-auto">
                    <Camera className="h-6 w-6 text-blue-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Smart Capture</h3>
                  <p className="text-gray-600 text-sm">
                    Advanced guidance for optimal image quality with real-time feedback
                  </p>
                </div>
                <div className="bg-white/60 backdrop-blur-lg rounded-2xl p-6 border border-gray-100 shadow-sm">
                  <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4 mx-auto">
                    <Eye className="h-6 w-6 text-purple-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Analysis</h3>
                  <p className="text-gray-600 text-sm">
                    Detect 6+ skin conditions with explainable AI and confidence scoring
                  </p>
                </div>
                <div className="bg-white/60 backdrop-blur-lg rounded-2xl p-6 border border-gray-100 shadow-sm">
                  <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4 mx-auto">
                    <Users className="h-6 w-6 text-green-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Expert Guidance</h3>
                  <p className="text-gray-600 text-sm">
                    Personalized recommendations and optional clinician review
                  </p>
                </div>
              </div>

              {/* CTA Button */}
              <button
                onClick={handleStartAnalysis}
                className="inline-flex items-center px-8 py-4 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold text-lg shadow-lg hover:shadow-xl transition-all transform hover:scale-105"
              >
                <Camera className="h-5 w-5 mr-3" />
                Start Skin Analysis
              </button>

              {/* Privacy Notice */}
              <p className="text-sm text-gray-500 mt-6 max-w-2xl mx-auto">
                Your privacy is protected. Images are processed securely and you control data usage. 
                This tool provides cosmetic recommendations and is not a medical diagnostic device.
              </p>
            </div>
          </div>
        )}

        {currentStep === 'capture' && (
          <CameraCapture
            onImageCapture={handleImageCapture}
            onBack={handleBackToWelcome}
          />
        )}

        {currentStep === 'results' && results && uploadedImage && (
          <ResultsDisplay
            results={results}
            originalImage={uploadedImage}
            onRetakePhoto={handleRetakePhoto}
            onBackToWelcome={handleBackToWelcome}
          />
        )}

        {currentStep === 'admin' && (
          <AdminDashboard onBack={handleBackToWelcome} />
        )}
      </main>

      {/* Consent Modal */}
      {showConsent && (
        <ConsentModal
          onAccept={handleConsentAccept}
          onDecline={() => setShowConsent(false)}
        />
      )}
    </div>
  );
};

export default App;