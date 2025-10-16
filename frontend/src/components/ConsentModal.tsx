import React, { useState } from 'react';
import { Shield, Check, X, Info } from 'lucide-react';

interface ConsentModalProps {
  onAccept: () => void;
  onDecline: () => void;
}

const ConsentModal: React.FC<ConsentModalProps> = ({ onAccept, onDecline }) => {
  const [allowDataCollection, setAllowDataCollection] = useState(false);
  const [agreeToTerms, setAgreeToTerms] = useState(false);

  const handleAccept = () => {
    if (!agreeToTerms) {
      alert('Please agree to the terms and conditions to proceed.');
      return;
    }
    onAccept();
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center mb-6">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mr-4">
              <Shield className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Privacy & Consent</h2>
              <p className="text-gray-600">Your data privacy is our priority</p>
            </div>
          </div>

          {/* Privacy Information */}
          <div className="mb-6 space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 mb-2">How we protect your privacy:</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Images are processed with advanced encryption (AES-256)</li>
                <li>• All data transmission uses secure TLS protocols</li>
                <li>• Personal identifiers are anonymized and hashed</li>
                <li>• You control how long your data is stored</li>
              </ul>
            </div>

            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">What happens to your image:</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• EXIF metadata is automatically stripped for privacy</li>
                <li>• Images are processed by our AI model to detect skin conditions</li>
                <li>• Analysis results are generated and returned to you</li>
                <li>• Images are automatically deleted after 24 hours unless you opt-in below</li>
              </ul>
            </div>
          </div>

          {/* Consent Options */}
          <div className="mb-6 space-y-4">
            <h3 className="font-semibold text-gray-900">Data Usage Preferences</h3>
            
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <label className="flex items-start cursor-pointer">
                <input
                  type="checkbox"
                  checked={allowDataCollection}
                  onChange={(e) => setAllowDataCollection(e.target.checked)}
                  className="mt-1 mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <div className="font-medium text-gray-900">
                    Allow anonymous data collection for model improvement (Optional)
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    Help us improve our AI model by contributing your anonymized images and analysis results to our training dataset. 
                    Your images will be stripped of all personal information and used only for research and model enhancement.
                  </div>
                  <div className="text-xs text-blue-600 mt-2">
                    <Info className="h-3 w-3 inline mr-1" />
                    You can revoke this consent at any time by contacting support
                  </div>
                </div>
              </label>
            </div>

            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <label className="flex items-start cursor-pointer">
                <input
                  type="checkbox"
                  checked={agreeToTerms}
                  onChange={(e) => setAgreeToTerms(e.target.checked)}
                  className="mt-1 mr-3 h-4 w-4 text-amber-600 focus:ring-amber-500 border-gray-300 rounded"
                />
                <div>
                  <div className="font-medium text-amber-900">
                    I agree to the Terms of Service and Privacy Policy (Required)
                  </div>
                  <div className="text-sm text-amber-800 mt-1">
                    By proceeding, you acknowledge that this tool provides cosmetic analysis and recommendations only, 
                    and is not a substitute for professional medical advice.
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Medical Disclaimer */}
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="font-semibold text-red-900 mb-2">Important Medical Disclaimer</h3>
            <p className="text-sm text-red-800">
              <strong>This is not a medical diagnostic device.</strong> SkinScan Pro provides cosmetic analysis and 
              educational information only. For any medical concerns, suspicious lesions, or persistent skin issues, 
              please consult with a qualified dermatologist or healthcare provider. Do not rely on this analysis 
              for medical decision-making.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 pt-4 border-t border-gray-200">
            <button
              onClick={onDecline}
              className="flex items-center justify-center px-6 py-3 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-full font-semibold transition-colors"
            >
              <X className="h-4 w-4 mr-2" />
              Decline
            </button>
            <button
              onClick={handleAccept}
              disabled={!agreeToTerms}
              className="flex items-center justify-center px-6 py-3 text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg rounded-full font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Check className="h-4 w-4 mr-2" />
              Accept & Continue
            </button>
          </div>

          {/* Additional Links */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex flex-wrap justify-center gap-4 text-sm text-gray-600">
              <a href="#" className="hover:text-gray-900 underline">Privacy Policy</a>
              <a href="#" className="hover:text-gray-900 underline">Terms of Service</a>
              <a href="#" className="hover:text-gray-900 underline">Cookie Policy</a>
              <a href="#" className="hover:text-gray-900 underline">Contact Support</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConsentModal;