import React, { useState, useRef, useEffect } from 'react';
import { ArrowLeft, RotateCcw, Download, Info, Eye, EyeOff, Zap } from 'lucide-react';
import { PredictionResults } from '../AppRoutes';

interface ResultsDisplayProps {
  results: PredictionResults;
  originalImage: string;
  onRetakePhoto: () => void;
  onBackToWelcome: () => void;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  results,
  originalImage,
  onRetakePhoto,
  onBackToWelcome
}) => {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6);
  const [selectedCondition, setSelectedCondition] = useState<string | null>(null);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Render combined image with heatmap overlay
  useEffect(() => {
    if (!canvasRef.current || !results.heatmap_b64) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const originalImg = new Image();
    originalImg.onload = () => {
      canvas.width = originalImg.width;
      canvas.height = originalImg.height;
      
      // Draw original image
      ctx.drawImage(originalImg, 0, 0);
      
      if (showHeatmap && results.heatmap_b64) {
        // Create heatmap image
        const heatmapImg = new Image();
        heatmapImg.onload = () => {
          // Apply heatmap with opacity
          ctx.globalAlpha = heatmapOpacity;
          ctx.globalCompositeOperation = 'multiply';
          ctx.drawImage(heatmapImg, 0, 0, canvas.width, canvas.height);
          
          // Reset composite operation
          ctx.globalCompositeOperation = 'source-over';
          ctx.globalAlpha = 1.0;
          
          // Draw landmarks if available
          drawLandmarks(ctx, canvas.width, canvas.height);
        };
        heatmapImg.src = `data:image/png;base64,${results.heatmap_b64}`;
      } else {
        drawLandmarks(ctx, canvas.width, canvas.height);
      }
    };
    originalImg.src = originalImage;
  }, [originalImage, results.heatmap_b64, showHeatmap, heatmapOpacity, selectedZone]);

  const drawLandmarks = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (!results.landmarks) return;

    results.landmarks.forEach((landmark: any) => {
      const x = landmark.x * width;
      const y = landmark.y * height;
      
      // Color code by zone
      const zoneColors: Record<string, string> = {
        'forehead': '#3B82F6',
        'left_eye': '#10B981',
        'right_eye': '#10B981',
        'nose': '#F59E0B',
        'left_cheek': '#EF4444',
        'right_cheek': '#EF4444',
        'mouth': '#8B5CF6',
        'chin': '#6B7280',
        'other': '#9CA3AF'
      };
      
      const color = zoneColors[landmark.zone] || '#9CA3AF';
      const isSelected = selectedZone === landmark.zone;
      
      ctx.beginPath();
      ctx.arc(x, y, isSelected ? 4 : 2, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      
      if (isSelected) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  const getConditionColor = (condition: string, score: number) => {
    const intensity = Math.min(score, 1.0);
    const baseColors: Record<string, string> = {
      'acne': '239, 68, 68',        // red
      'hyperpigmentation': '139, 92, 246', // purple
      'redness': '245, 158, 11',    // amber
      'dehydration': '59, 130, 246', // blue
      'pore_size': '16, 185, 129',  // emerald
      'fine_lines': '107, 114, 128' // gray
    };
    
    const color = baseColors[condition] || '156, 163, 175';
    return `rgba(${color}, ${0.1 + intensity * 0.4})`;
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-red-600 bg-red-50 border-red-200';
    if (score >= 0.5) return 'text-amber-600 bg-amber-50 border-amber-200';
    if (score >= 0.3) return 'text-blue-600 bg-blue-50 border-blue-200';
    return 'text-green-600 bg-green-50 border-green-200';
  };

  const downloadReport = () => {
    // Create a comprehensive report
    const reportData = {
      timestamp: results.timestamp || new Date().toISOString(),
      confidence: results.confidence || 'medium',
      predictions: results.predictions,
      explanations: results.explanations || [],
      zone_scores: results.zone_scores || {}
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `skin-analysis-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onBackToWelcome}
          className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Home
        </button>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={downloadReport}
            className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            <Download className="h-4 w-4 mr-2" />
            Download Report
          </button>
          <button
            onClick={onRetakePhoto}
            className="flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:shadow-lg"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            New Analysis
          </button>
        </div>
      </div>

      {/* Main Results */}
      <div className="grid lg:grid-cols-2 gap-8 mb-8">
        {/* Image with Overlay */}
        <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
          <div className="p-4 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Analysis Visualization</h3>
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={`flex items-center px-3 py-1 text-sm rounded-full transition-colors ${
                    showHeatmap 
                      ? 'bg-blue-100 text-blue-700' 
                      : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {showHeatmap ? <Eye className="h-4 w-4 mr-1" /> : <EyeOff className="h-4 w-4 mr-1" />}
                  Heatmap
                </button>
              </div>
            </div>
            
            {showHeatmap && (
              <div className="mt-3">
                <label className="text-sm text-gray-600">Overlay Opacity</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  className="w-full mt-1"
                />
              </div>
            )}
          </div>
          
          <div className="relative">
            <canvas
              ref={canvasRef}
              className="w-full h-auto cursor-crosshair"
              onClick={(e) => {
                // Handle zone selection on click
                const rect = canvasRef.current?.getBoundingClientRect();
                if (!rect) return;
                
                const x = (e.clientX - rect.left) / rect.width;
                const y = (e.clientY - rect.top) / rect.height;
                
                // Find closest landmark
                let closestZone = null;
                let minDistance = Infinity;
                
                results.landmarks?.forEach((landmark: any) => {
                  const distance = Math.sqrt(
                    Math.pow(landmark.x - x, 2) + Math.pow(landmark.y - y, 2)
                  );
                  if (distance < minDistance && distance < 0.1) {
                    minDistance = distance;
                    closestZone = landmark.zone;
                  }
                });
                
                setSelectedZone(closestZone);
              }}
            />
          </div>
        </div>

        {/* Analysis Results */}
        <div className="bg-white rounded-2xl shadow-lg">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Skin Analysis Results</h3>
              <div className="flex items-center">
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  results.confidence === 'high' ? 'bg-green-100 text-green-700' :
                  results.confidence === 'medium' ? 'bg-amber-100 text-amber-700' :
                  'bg-red-100 text-red-700'
                }`}>
                  {(results.confidence || 'medium').charAt(0).toUpperCase() + (results.confidence || 'medium').slice(1)} Confidence
                </div>
              </div>
            </div>

            {/* Condition Scores */}
            <div className="space-y-4 mb-6">
              <h4 className="font-medium text-gray-900">Detected Conditions</h4>
              {Object.entries(results.predictions).map(([condition, score]: [string, any]) => (
                <div
                  key={condition}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedCondition === condition 
                      ? 'ring-2 ring-blue-500 border-blue-200' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  style={{ backgroundColor: getConditionColor(condition, score as number) }}
                  onClick={() => setSelectedCondition(selectedCondition === condition ? null : condition)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="font-medium text-gray-900 capitalize">
                        {condition.replace('_', ' ')}
                      </span>
                      {(score as number) > 0.5 && <Zap className="h-4 w-4 ml-2 text-amber-500" />}
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600">{Math.round((score as number) * 100)}%</span>
                      <div className={`px-2 py-1 rounded text-xs border ${getScoreColor(score as number)}`}>
                        {(score as number) >= 0.7 ? 'High' : (score as number) >= 0.5 ? 'Moderate' : (score as number) >= 0.3 ? 'Mild' : 'Low'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all"
                        style={{ width: `${Math.min((score as number) * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Zone Information */}
            {selectedZone && results.zone_scores?.[selectedZone] !== undefined && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <h4 className="font-medium text-blue-900 mb-2">
                  {selectedZone.charAt(0).toUpperCase() + selectedZone.slice(1).replace('_', ' ')} Zone
                </h4>
                <div className="text-sm text-blue-700">
                  Zone Score: {Math.round((results.zone_scores?.[selectedZone] || 0) * 100)}%
                </div>
                <p className="text-sm text-blue-600 mt-1">
                  Click on other landmark zones to compare regional analysis.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Detailed Explanations */}
      <div className="bg-white rounded-2xl shadow-lg">
        <div className="p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Detailed Analysis & Recommendations</h3>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.explanations?.map((explanation: any, index: number) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-900 capitalize">
                    {explanation.condition.replace('_', ' ')}
                  </h4>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    explanation.severity === 'high' ? 'bg-red-100 text-red-700' :
                    explanation.severity === 'moderate' ? 'bg-amber-100 text-amber-700' :
                    'bg-blue-100 text-blue-700'
                  }`}>
                    {explanation.severity}
                  </span>
                </div>
                
                <p className="text-gray-600 text-sm mb-4">
                  {explanation.explanation}
                </p>
                
                <div>
                  <h5 className="font-medium text-gray-900 text-sm mb-2">Recommendations:</h5>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {explanation.recommendations.slice(0, 3).map((rec: any, i: number) => (
                      <li key={i} className="flex items-start">
                        <span className="text-blue-500 mr-2">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>

          {/* Medical Disclaimer */}
          <div className="mt-8 bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="flex items-start">
              <Info className="h-5 w-5 text-amber-600 mt-0.5 mr-3 flex-shrink-0" />
              <div>
                <h4 className="text-amber-800 font-medium mb-1">Important Notice</h4>
                <p className="text-amber-700 text-sm">
                  This analysis is for cosmetic and educational purposes only and should not be used as a substitute 
                  for professional medical advice. If you notice sudden changes, persistent issues, or have concerns 
                  about any skin conditions, please consult with a qualified dermatologist or healthcare provider.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;