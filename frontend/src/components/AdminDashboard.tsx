import React, { useState, useEffect } from 'react';
import { ArrowLeft, Activity, BarChart3, Users, Database, CheckCircle, AlertCircle, Clock } from 'lucide-react';

interface ModelStatus {
  model_loaded: boolean;
  model_version: string;
  supported_conditions: string[];
  last_updated: string;
  accuracy_metrics: {
    overall: number;
    per_condition: Record<string, number>;
    per_skin_tone: Record<string, number>;
  };
}

interface AdminDashboardProps {
  onBack: () => void;
}

const AdminDashboard: React.FC<AdminDashboardProps> = ({ onBack }) => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/v1/model/status');
      
      if (!response.ok) {
        throw new Error('Failed to fetch model status');
      }
      
      const data = await response.json();
      setModelStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 0.85) return 'text-green-600 bg-green-50';
    if (accuracy >= 0.75) return 'text-amber-600 bg-amber-50';
    return 'text-red-600 bg-red-50';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading model status...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto px-4">
        <div className="mb-6">
          <button
            onClick={onBack}
            className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </button>
        </div>
        
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error Loading Dashboard</h3>
          <p className="text-red-700 mb-4">{error}</p>
          <button
            onClick={fetchModelStatus}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <button
            onClick={onBack}
            className="flex items-center text-gray-600 hover:text-gray-900 transition-colors mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </button>
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="text-gray-600">Model performance and system status</p>
        </div>
        
        <button
          onClick={fetchModelStatus}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Activity className="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>

      {/* Status Cards */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            {modelStatus?.model_loaded ? (
              <CheckCircle className="h-8 w-8 text-green-500" />
            ) : (
              <AlertCircle className="h-8 w-8 text-red-500" />
            )}
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Model Status</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelStatus?.model_loaded ? 'Active' : 'Inactive'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <BarChart3 className="h-8 w-8 text-blue-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Overall Accuracy</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelStatus?.accuracy_metrics.overall ? 
                  `${Math.round(modelStatus.accuracy_metrics.overall * 100)}%` : 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <Database className="h-8 w-8 text-purple-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Supported Conditions</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelStatus?.supported_conditions.length || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-amber-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Model Version</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelStatus?.model_version || 'Unknown'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="grid lg:grid-cols-2 gap-8 mb-8">
        {/* Per-Condition Accuracy */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Condition-Specific Accuracy</h3>
            <p className="text-gray-600">Performance metrics for each skin condition</p>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {modelStatus?.accuracy_metrics.per_condition ? 
                Object.entries(modelStatus.accuracy_metrics.per_condition).map(([condition, accuracy]) => (
                  <div key={condition} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="font-medium text-gray-900 capitalize">
                        {condition.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-24 h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-2 bg-gradient-to-r from-blue-500 to-green-500 rounded-full"
                          style={{ width: `${accuracy * 100}%` }}
                        />
                      </div>
                      <span className={`px-2 py-1 rounded text-sm font-medium ${getAccuracyColor(accuracy)}`}>
                        {Math.round(accuracy * 100)}%
                      </span>
                    </div>
                  </div>
                )) : (
                  <p className="text-gray-500 text-center py-4">No condition data available</p>
                )}
            </div>
          </div>
        </div>

        {/* Per-Skin-Tone Performance */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Skin Tone Performance</h3>
            <p className="text-gray-600">Fairness metrics across different skin tones</p>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {modelStatus?.accuracy_metrics.per_skin_tone ? 
                Object.entries(modelStatus.accuracy_metrics.per_skin_tone).map(([tone, accuracy]) => (
                  <div key={tone} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="font-medium text-gray-900">
                        {tone.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-24 h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-2 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
                          style={{ width: `${accuracy * 100}%` }}
                        />
                      </div>
                      <span className={`px-2 py-1 rounded text-sm font-medium ${getAccuracyColor(accuracy)}`}>
                        {Math.round(accuracy * 100)}%
                      </span>
                    </div>
                  </div>
                )) : (
                  <p className="text-gray-500 text-center py-4">No skin tone data available</p>
                )}
            </div>
          </div>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">System Information</h3>
          <p className="text-gray-600">Current model configuration and metadata</p>
        </div>
        <div className="p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Model Details</h4>
              <dl className="space-y-2">
                <div className="flex justify-between">
                  <dt className="text-sm text-gray-600">Version:</dt>
                  <dd className="text-sm font-medium text-gray-900">
                    {modelStatus?.model_version || 'Unknown'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-sm text-gray-600">Last Updated:</dt>
                  <dd className="text-sm font-medium text-gray-900">
                    {modelStatus?.last_updated ? formatDate(modelStatus.last_updated) : 'Unknown'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-sm text-gray-600">Status:</dt>
                  <dd className="text-sm font-medium">
                    <span className={`px-2 py-1 rounded text-xs ${
                      modelStatus?.model_loaded ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {modelStatus?.model_loaded ? 'Loaded' : 'Not Loaded'}
                    </span>
                  </dd>
                </div>
              </dl>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-3">Supported Conditions</h4>
              <div className="flex flex-wrap gap-2">
                {modelStatus?.supported_conditions.map((condition) => (
                  <span
                    key={condition}
                    className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full capitalize"
                  >
                    {condition.replace('_', ' ')}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Notes */}
      <div className="mt-8 bg-amber-50 border border-amber-200 rounded-lg p-4">
        <h4 className="font-medium text-amber-900 mb-2">Performance Notes</h4>
        <ul className="text-sm text-amber-800 space-y-1">
          <li>• Accuracy metrics are calculated on validation datasets</li>
          <li>• Skin tone performance helps ensure fairness across different demographics</li>
          <li>• Model performance may vary on real-world data compared to validation metrics</li>
          <li>• Regular model updates and retraining help improve accuracy over time</li>
        </ul>
      </div>
    </div>
  );
};

export default AdminDashboard;