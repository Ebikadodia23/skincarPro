import React from 'react';
import { Activity } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ message = "Processing..." }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center">
      <div className="text-center">
        <div className="relative">
          {/* Animated background circles */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-32 h-32 border-4 border-blue-100 rounded-full animate-ping"></div>
          </div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-24 h-24 border-4 border-purple-200 rounded-full animate-ping animation-delay-150"></div>
          </div>
          
          {/* Central spinner */}
          <div className="relative w-20 h-20 mx-auto mb-6">
            <div className="absolute inset-0 border-4 border-gray-200 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-transparent border-t-blue-600 border-r-purple-600 rounded-full animate-spin"></div>
            
            {/* Icon in center */}
            <div className="absolute inset-0 flex items-center justify-center">
              <Activity className="h-8 w-8 text-blue-600 animate-pulse" />
            </div>
          </div>
        </div>
        
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">{message}</h2>
        <p className="text-gray-600 mb-6">
          Our AI is analyzing your skin with advanced computer vision
        </p>
        
        {/* Progress indicators */}
        <div className="max-w-md mx-auto">
          <div className="flex justify-between text-sm text-gray-500 mb-2">
            <span>Processing image...</span>
            <span>Computing features...</span>
            <span>Generating insights...</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full animate-pulse" 
                 style={{ width: '75%' }}>
            </div>
          </div>
        </div>
        
        <div className="mt-8 text-sm text-gray-500">
          This usually takes 5-10 seconds
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner;