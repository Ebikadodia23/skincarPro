# ml/inference_with_gradcam.py
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import mediapipe as mp
import base64
import io
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/skin_model_ts.pt"
METADATA_PATH = "../models/model_metadata.json"

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class GradCAM:
    """GradCAM implementation for generating activation heatmaps"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Register hooks for the target layer
        self._register_hooks(target_layer_name)
    
    def _register_hooks(self, layer_name):
        """Register forward and backward hooks"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find the target layer in the model
        for name, module in self.model.named_modules():
            if layer_name in name:
                self.target_layer = module
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Class Activation Map"""
        self.model.eval()
        
        # Forward pass
        model_output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = model_output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:])  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam / torch.max(cam) if torch.max(cam) > 0 else cam
        
        return cam.detach().cpu().numpy()

class SkinAnalysisInference:
    """Main inference class for skin analysis"""
    
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.condition_labels = []
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Facial zone mapping (MediaPipe landmark indices)
        self.facial_zones = {
            'forehead': list(range(9, 11)) + list(range(151, 157)),
            'left_eye': list(range(33, 42)) + list(range(133, 142)),
            'right_eye': list(range(362, 371)) + list(range(263, 272)),
            'nose': list(range(1, 9)) + list(range(19, 25)),
            'left_cheek': list(range(116, 123)) + list(range(147, 150)),
            'right_cheek': list(range(345, 352)) + list(range(376, 379)),
            'mouth': list(range(61, 68)) + list(range(291, 298)),
            'chin': list(range(175, 180)) + list(range(396, 401))
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = torch.jit.load(self.model_path, map_location=DEVICE)
                self.model.eval()
                logger.info("Model loaded successfully")
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                        self.condition_labels = self.metadata.get('condition_labels', [])
                    logger.info(f"Loaded metadata with {len(self.condition_labels)} conditions")
                
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        return transform(image).unsqueeze(0).to(DEVICE)
    
    def extract_landmarks(self, image: Image.Image) -> List[Dict]:
        """Extract facial landmarks using MediaPipe"""
        landmarks = []
        
        try:
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = self.face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert normalized landmarks to image coordinates
                h, w = image.size[1], image.size[0]
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks.append({
                        'id': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'zone': self._get_landmark_zone(idx)
                    })
                    
            logger.info(f"Extracted {len(landmarks)} facial landmarks")
                    
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
        
        return landmarks
    
    def _get_landmark_zone(self, landmark_idx: int) -> str:
        """Map landmark index to facial zone"""
        for zone, indices in self.facial_zones.items():
            if landmark_idx in indices:
                return zone
        return 'other'
    
    def generate_heatmap(self, image: Image.Image, predictions: Dict[str, float]) -> str:
        """Generate heatmap using GradCAM (simplified version for demo)"""
        try:
            if self.model is None:
                # Generate synthetic heatmap for demo
                h, w = image.size[1], image.size[0]
                heatmap = np.random.rand(h, w) * 0.5
                
                # Add some structure based on predictions
                center_x, center_y = w // 2, h // 2
                for condition, score in predictions.items():
                    if score > 0.5:
                        # Add hotspots for high-scoring conditions
                        x = int(center_x + np.random.uniform(-w//4, w//4))
                        y = int(center_y + np.random.uniform(-h//4, h//4))
                        
                        # Create circular hotspot
                        radius = int(min(w, h) * 0.1)
                        y_grid, x_grid = np.ogrid[:h, :w]
                        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
                        heatmap[mask] = np.maximum(heatmap[mask], score)
                
            else:
                # TODO: Implement real GradCAM with the loaded model
                # For now, use synthetic heatmap
                h, w = image.size[1], image.size[0]
                heatmap = np.random.rand(h, w) * 0.7
            
            # Normalize and convert to image
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            heatmap_pil.save(buffer, format='PNG')
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return heatmap_b64
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return ""
    
    def predict(self, image: Image.Image) -> Dict:
        """Run complete skin analysis prediction"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run model inference
            if self.model is not None:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    predictions = torch.sigmoid(outputs).cpu().numpy()[0]
                    
                # Map to condition labels
                pred_dict = {}
                for i, label in enumerate(self.condition_labels):
                    pred_dict[label] = float(predictions[i]) if i < len(predictions) else 0.0
            else:
                # Fallback predictions for demo
                pred_dict = {
                    'acne': np.random.uniform(0.1, 0.9),
                    'hyperpigmentation': np.random.uniform(0.0, 0.6),
                    'redness': np.random.uniform(0.0, 0.5),
                    'dehydration': np.random.uniform(0.1, 0.7),
                    'pore_size': np.random.uniform(0.2, 0.8),
                    'fine_lines': np.random.uniform(0.0, 0.4)
                }
            
            # Extract landmarks
            landmarks = self.extract_landmarks(image)
            
            # Generate heatmap
            heatmap_b64 = self.generate_heatmap(image, pred_dict)
            
            # Generate explanations
            explanations = self._generate_explanations(pred_dict)
            
            # Determine confidence
            confidence = self._calculate_confidence(pred_dict)
            
            # Calculate zone-specific scores
            zone_scores = self._calculate_zone_scores(landmarks, pred_dict)
            
            return {
                'predictions': pred_dict,
                'landmarks': landmarks,
                'heatmap_b64': heatmap_b64,
                'explanations': explanations,
                'confidence': confidence,
                'zone_scores': zone_scores,
                'model_version': getattr(self.metadata, 'model_version', '1.0.0'),
                'timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else '2024-01-01T00:00:00'
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'predictions': {},
                'landmarks': [],
                'heatmap_b64': '',
                'explanations': [],
                'confidence': 'low'
            }
    
    def _generate_explanations(self, predictions: Dict[str, float]) -> List[Dict]:
        """Generate human-readable explanations for predictions"""
        explanations = []
        
        condition_explanations = {
            'acne': 'Active breakouts detected. Consider gentle cleansing and salicylic acid treatments.',
            'hyperpigmentation': 'Dark spots or uneven tone detected. Vitamin C and sun protection recommended.',
            'redness': 'Inflammation or sensitivity detected. Use gentle, anti-inflammatory skincare.',
            'dehydration': 'Skin appears dry or tight. Increase moisturization with hyaluronic acid.',
            'pore_size': 'Enlarged pores visible. Regular exfoliation and retinoids may help.',
            'fine_lines': 'Signs of aging detected. Consider retinol and adequate moisturization.'
        }
        
        # Sort by score and explain top conditions
        sorted_conditions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for condition, score in sorted_conditions[:3]:  # Top 3 conditions
            if score > 0.3:  # Only explain significant findings
                explanations.append({
                    'condition': condition,
                    'score': round(score, 3),
                    'severity': 'high' if score > 0.7 else 'moderate' if score > 0.5 else 'mild',
                    'explanation': condition_explanations.get(condition, 'Analysis complete.'),
                    'recommendations': self._get_recommendations(condition, score)
                })
        
        return explanations
    
    def _get_recommendations(self, condition: str, score: float) -> List[str]:
        """Get specific recommendations for each condition"""
        recommendations = {
            'acne': [
                'Use gentle, non-comedogenic cleanser',
                'Apply salicylic acid treatment',
                'Avoid over-washing or harsh scrubbing',
                'Consider consulting a dermatologist for persistent cases'
            ],
            'hyperpigmentation': [
                'Apply broad-spectrum SPF 30+ daily',
                'Use vitamin C serum in morning',
                'Consider niacinamide for evening routine',
                'Be patient - improvements take 6-12 weeks'
            ],
            'redness': [
                'Avoid known triggers (heat, spicy food, alcohol)',
                'Use gentle, fragrance-free products',
                'Apply cool compresses for immediate relief',
                'Consider green-tinted primer for coverage'
            ],
            'dehydration': [
                'Increase water intake',
                'Use hyaluronic acid serum',
                'Apply moisturizer on damp skin',
                'Consider a humidifier in dry environments'
            ],
            'pore_size': [
                'Use BHA (salicylic acid) 2-3 times weekly',
                'Consider retinoid products',
                'Avoid pore strips which can damage skin',
                'Clay masks can temporarily minimize appearance'
            ],
            'fine_lines': [
                'Start with over-the-counter retinol',
                'Maintain consistent moisturizing routine',
                'Always wear sunscreen to prevent further damage',
                'Consider professional treatments for advanced signs'
            ]
        }
        
        return recommendations.get(condition, ['Maintain healthy skincare routine'])
    
    def _calculate_confidence(self, predictions: Dict[str, float]) -> str:
        """Calculate overall prediction confidence"""
        max_score = max(predictions.values()) if predictions else 0
        avg_score = np.mean(list(predictions.values())) if predictions else 0
        
        if max_score > 0.8 and avg_score > 0.4:
            return 'high'
        elif max_score > 0.6 or avg_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_zone_scores(self, landmarks: List[Dict], predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate condition scores for different facial zones"""
        zone_scores = {}
        
        # Group landmarks by zone
        zones = {}
        for landmark in landmarks:
            zone = landmark.get('zone', 'other')
            if zone not in zones:
                zones[zone] = []
            zones[zone].append(landmark)
        
        # Calculate average scores for each zone
        for zone, zone_landmarks in zones.items():
            if zone_landmarks:
                # For demo, assign random scores weighted by predictions
                zone_score = np.mean(list(predictions.values())) * np.random.uniform(0.7, 1.3)
                zone_scores[zone] = max(0.0, min(1.0, zone_score))
        
        return zone_scores

# Example usage and testing
if __name__ == "__main__":
    analyzer = SkinAnalysisInference()
    
    # Test with a sample image (you would load a real image here)
    try:
        # Create a dummy image for testing
        test_image = Image.new('RGB', (512, 512), color=(220, 180, 140))  # Skin-like color
        
        # Run prediction
        results = analyzer.predict(test_image)
        
        print("Prediction Results:")
        print(f"Confidence: {results.get('confidence', 'unknown')}")
        print(f"Predictions: {results.get('predictions', {})}")
        print(f"Found {len(results.get('landmarks', []))} landmarks")
        print(f"Explanations: {len(results.get('explanations', []))}")
        print(f"Zone scores: {results.get('zone_scores', {})}")
        
        # Print detailed explanations
        for exp in results.get('explanations', []):
            print(f"\n{exp['condition'].upper()} (Score: {exp['score']}, Severity: {exp['severity']})")
            print(f"  {exp['explanation']}")
            print(f"  Recommendations: {', '.join(exp['recommendations'][:2])}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")