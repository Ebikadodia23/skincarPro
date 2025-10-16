import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Upload, RotateCcw, CheckCircle, AlertTriangle, ArrowLeft } from 'lucide-react';

interface CameraCaptureProps {
  onImageCapture: (blob: Blob, dataUrl: string) => void;
  onBack: () => void;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({ onImageCapture, onBack }) => {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [imageQuality, setImageQuality] = useState<{
    blur: number;
    lighting: 'good' | 'dark' | 'bright';
    distance: 'good' | 'close' | 'far';
  }>({ blur: 0, lighting: 'good', distance: 'good' });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize camera
  useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Real-time quality analysis
  useEffect(() => {
    if (!cameraReady || !videoRef.current || !canvasRef.current) return;

    const analyzeQuality = () => {
      const video = videoRef.current!;
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext('2d')!;

      canvas.width = video.videoWidth / 4; // Lower res for analysis
      canvas.height = video.videoHeight / 4;
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Simple blur detection using edge detection
      let edgeCount = 0;
      const pixels = imageData.data;
      
      for (let i = 0; i < pixels.length - 4; i += 4) {
        const diff = Math.abs(pixels[i] - pixels[i + 4]);
        if (diff > 30) edgeCount++;
      }
      
      const blurScore = edgeCount / (canvas.width * canvas.height) * 100;
      
      // Lighting analysis (average brightness)
      let brightness = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        brightness += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
      }
      brightness /= (pixels.length / 4);
      
      const lighting = brightness < 100 ? 'dark' : brightness > 200 ? 'bright' : 'good';
      
      setImageQuality({
        blur: blurScore,
        lighting,
        distance: 'good' // Would need more sophisticated analysis
      });
    };

    const interval = setInterval(analyzeQuality, 1000);
    return () => clearInterval(interval);
  }, [cameraReady]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });
      
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.onloadedmetadata = () => {
          setCameraReady(true);
        };
      }
    } catch (error) {
      console.error('Camera access failed:', error);
      alert('Camera access is required for skin analysis. Please allow camera permissions.');
    }
  };

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Set canvas dimensions to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0);

    // Convert to blob and data URL
    canvas.toBlob((blob) => {
      if (blob) {
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        setCapturedImage(dataUrl);
      }
    }, 'image/jpeg', 0.9);
  }, []);

  const confirmCapture = () => {
    if (!canvasRef.current || !capturedImage) return;

    canvasRef.current.toBlob((blob) => {
      if (blob) {
        onImageCapture(blob, capturedImage);
      }
    }, 'image/jpeg', 0.9);
  };

  const retakePhoto = () => {
    setCapturedImage(null);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setCapturedImage(dataUrl);
      
      // Convert data URL to blob
      fetch(dataUrl)
        .then(res => res.blob())
        .then(blob => {
          // Auto-confirm file uploads
          onImageCapture(blob, dataUrl);
        });
    };
    reader.readAsDataURL(file);
  };

  const getQualityIndicator = () => {
    const issues = [];
    
    if (imageQuality.blur < 15) issues.push('Image may be blurry - hold steady');
    if (imageQuality.lighting === 'dark') issues.push('Lighting too dark - move to brighter area');
    if (imageQuality.lighting === 'bright') issues.push('Too bright - avoid direct light');
    if (imageQuality.distance === 'close') issues.push('Too close - move camera back');
    if (imageQuality.distance === 'far') issues.push('Too far - move camera closer');
    
    return issues;
  };

  const qualityIssues = getQualityIndicator();
  const isGoodQuality = qualityIssues.length === 0 && imageQuality.blur > 15;

  return (
    <div className="max-w-4xl mx-auto px-4">
      <div className="mb-6">
        <button
          onClick={onBack}
          className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </button>
      </div>

      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Capture Your Image</h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Position your face or the area you'd like to analyze within the oval guide. 
          Ensure good lighting and hold the camera steady for the best results.
        </p>
      </div>

      {!capturedImage ? (
        <div className="relative bg-white rounded-2xl shadow-lg overflow-hidden">
          {/* Camera Feed */}
          <div className="relative aspect-video bg-gray-900 flex items-center justify-center">
            {cameraReady ? (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />
                {/* Overlay Guide */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-80 h-96 border-4 border-white/70 rounded-full border-dashed animate-pulse">
                    <div className="w-full h-full bg-white/10 rounded-full flex items-center justify-center">
                      <p className="text-white text-sm font-medium">Position face here</p>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-white">
                <Camera className="h-16 w-16 mx-auto mb-4 animate-pulse" />
                <p className="text-lg">Initializing camera...</p>
              </div>
            )}
          </div>

          {/* Quality Indicators */}
          <div className="bg-gray-50 p-4 border-t">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                {isGoodQuality ? (
                  <>
                    <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                    <span className="text-green-700 font-medium">Ready to capture</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-5 w-5 text-amber-500 mr-2" />
                    <span className="text-amber-700 font-medium">Adjust for better quality</span>
                  </>
                )}
              </div>
              <div className="text-sm text-gray-500">
                Quality: {Math.round(imageQuality.blur)}%
              </div>
            </div>

            {qualityIssues.length > 0 && (
              <div className="bg-amber-50 rounded-lg p-3 mb-4">
                <ul className="text-sm text-amber-700 space-y-1">
                  {qualityIssues.map((issue, index) => (
                    <li key={index}>• {issue}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Capture Controls */}
            <div className="flex items-center justify-center space-x-4">
              <button
                onClick={captureImage}
                disabled={!cameraReady}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-full hover:shadow-lg transition-all disabled:opacity-50"
              >
                <Camera className="h-5 w-5 mr-2" />
                Capture Photo
              </button>
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center px-6 py-3 bg-gray-100 text-gray-700 font-semibold rounded-full hover:bg-gray-200 transition-all"
              >
                <Upload className="h-5 w-5 mr-2" />
                Upload Image
              </button>
            </div>
          </div>

          {/* Hidden canvas for image capture */}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      ) : (
        /* Image Preview */
        <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
          <div className="aspect-video bg-gray-100 flex items-center justify-center">
            <img
              src={capturedImage}
              alt="Captured image"
              className="max-w-full max-h-full object-contain"
            />
          </div>
          
          <div className="p-6 bg-gray-50">
            <div className="text-center">
              <p className="text-gray-600 mb-6">
                Image captured successfully. Proceed with analysis or retake if needed.
              </p>
              
              <div className="flex items-center justify-center space-x-4">
                <button
                  onClick={confirmCapture}
                  className="flex items-center px-6 py-3 bg-gradient-to-r from-green-600 to-blue-600 text-white font-semibold rounded-full hover:shadow-lg transition-all"
                >
                  <CheckCircle className="h-5 w-5 mr-2" />
                  Analyze Image
                </button>
                
                <button
                  onClick={retakePhoto}
                  className="flex items-center px-6 py-3 bg-gray-100 text-gray-700 font-semibold rounded-full hover:bg-gray-200 transition-all"
                >
                  <RotateCcw className="h-5 w-5 mr-2" />
                  Retake Photo
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
};

export default CameraCapture;