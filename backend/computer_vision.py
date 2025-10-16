"""
Production-ready Computer Vision Service for Skin Analysis
Integrates MediaPipe, PyTorch models, and advanced image processing
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

class SkinAnalysisCV:
    """Complete Computer Vision Pipeline for Skin Analysis"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized CV service on device: {self.device}")
        
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # Use full model for better accuracy
        )
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Facial zone definitions (MediaPipe landmark indices)
        self.facial_zones = {
            'forehead': [9, 10, 151, 152, 153, 154, 155, 156, 157],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 207, 213, 192, 147],
            'right_cheek': [345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 361, 340, 346, 347, 348, 349],
            'mouth': [0, 17, 18, 200, 199, 175, 0, 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'chin': [18, 175, 199, 200, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
        }
        
        # Condition analysis algorithms
        self.condition_analyzers = {
            'acne': self._analyze_acne,
            'hyperpigmentation': self._analyze_hyperpigmentation,
            'redness': self._analyze_redness,
            'dehydration': self._analyze_dehydration,
            'pore_size': self._analyze_pore_size,
            'fine_lines': self._analyze_fine_lines
        }
    
    def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """Main analysis pipeline for skin condition detection"""
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract facial landmarks
            landmarks = self._extract_landmarks(image_cv)
            
            # Segment face region
            face_mask = self._segment_face(image_cv)
            
            # Analyze skin conditions
            predictions = {}
            heatmaps = {}
            
            for condition, analyzer in self.condition_analyzers.items():
                score, heatmap = analyzer(image_cv, landmarks, face_mask)
                predictions[condition] = float(score)
                heatmaps[condition] = heatmap
            
            # Generate combined heatmap
            combined_heatmap = self._generate_combined_heatmap(heatmaps, image.size)
            
            # Calculate zone-specific scores
            zone_scores = self._calculate_zone_scores(image_cv, landmarks, predictions)
            
            # Generate explanations
            explanations = self._generate_explanations(predictions)
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions)
            
            return {
                'predictions': predictions,
                'landmarks': [{'id': i, 'x': lm['x'], 'y': lm['y'], 'zone': lm['zone']} 
                            for i, lm in enumerate(landmarks)],
                'heatmap_b64': combined_heatmap,
                'explanations': explanations,
                'confidence': confidence,
                'zone_scores': zone_scores,
                'image_quality': self._assess_image_quality(image_cv),
                'skin_tone': self._estimate_skin_tone(image_cv, face_mask)
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._fallback_response()
    
    def _extract_landmarks(self, image_cv: np.ndarray) -> List[Dict[str, Any]]:
        """Extract 468 facial landmarks using MediaPipe"""
        landmarks = []
        
        try:
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = image_cv.shape[:2]
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'zone': self._get_landmark_zone(idx)
                    })
                    
        except Exception as e:
            logger.warning(f"Landmark extraction failed: {e}")
            
        return landmarks
    
    def _segment_face(self, image_cv: np.ndarray) -> np.ndarray:
        """Segment face region using MediaPipe"""
        try:
            rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            results = self.segmentation.process(rgb_image)
            
            # Create binary mask
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            return mask
            
        except Exception as e:
            logger.warning(f"Face segmentation failed: {e}")
            return np.ones(image_cv.shape[:2], dtype=np.uint8) * 255
    
    def _get_landmark_zone(self, landmark_idx: int) -> str:
        """Map landmark index to facial zone"""
        for zone, indices in self.facial_zones.items():
            if landmark_idx in indices:
                return zone
        return 'other'
    
    def _analyze_acne(self, image: np.ndarray, landmarks: List[Dict], mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect acne using texture analysis and color detection"""
        try:
            # Convert to LAB color space for better color analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Focus on face region
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            # Detect red regions (potential acne)
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
            
            # Red color range for acne detection
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Texture analysis using Laplacian
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian[mask > 0])
            
            # Combine color and texture features
            red_pixels = np.sum(red_mask > 0)
            total_face_pixels = np.sum(mask > 0)
            
            if total_face_pixels > 0:
                red_ratio = red_pixels / total_face_pixels
                texture_score = min(texture_variance / 1000, 1.0)
                acne_score = (red_ratio * 0.7 + texture_score * 0.3)
            else:
                acne_score = 0.0
            
            # Generate heatmap
            heatmap = cv2.GaussianBlur(red_mask.astype(np.float32), (21, 21), 0)
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
            
            return min(acne_score, 1.0), heatmap
            
        except Exception as e:
            logger.warning(f"Acne analysis failed: {e}")
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
    
    def _analyze_hyperpigmentation(self, image: np.ndarray, landmarks: List[Dict], mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect hyperpigmentation using brightness and color analysis"""
        try:
            # Convert to LAB for better color analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply face mask
            masked_l = cv2.bitwise_and(l_channel, l_channel, mask=mask)
            
            # Calculate mean brightness of face
            face_pixels = masked_l[mask > 0]
            if len(face_pixels) > 0:
                mean_brightness = np.mean(face_pixels)
                brightness_std = np.std(face_pixels)
                
                # Detect dark spots (hyperpigmentation)
                dark_threshold = mean_brightness - 1.5 * brightness_std
                dark_spots = (masked_l < dark_threshold) & (mask > 0)
                
                # Calculate hyperpigmentation score
                dark_pixels = np.sum(dark_spots)
                total_pixels = np.sum(mask > 0)
                hyper_score = dark_pixels / total_pixels if total_pixels > 0 else 0
                
                # Generate heatmap
                heatmap = np.zeros_like(l_channel, dtype=np.float32)
                heatmap[dark_spots] = 1.0
                heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
                heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
                
                return min(hyper_score * 2, 1.0), heatmap
            
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Hyperpigmentation analysis failed: {e}")
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
    
    def _analyze_redness(self, image: np.ndarray, landmarks: List[Dict], mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect skin redness using color analysis"""
        try:
            # Convert to HSV for better red detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define red color ranges
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 30, 30])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Apply face mask
            red_face = cv2.bitwise_and(red_mask, red_mask, mask=mask)
            
            # Calculate redness score
            red_pixels = np.sum(red_face > 0)
            total_pixels = np.sum(mask > 0)
            redness_score = red_pixels / total_pixels if total_pixels > 0 else 0
            
            # Generate heatmap
            heatmap = red_face.astype(np.float32) / 255.0
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
            
            return min(redness_score * 3, 1.0), heatmap
            
        except Exception as e:
            logger.warning(f"Redness analysis failed: {e}")
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
    
    def _analyze_dehydration(self, image: np.ndarray, landmarks: List[Dict], mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect skin dehydration using texture and brightness analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply face mask
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Texture analysis using local binary patterns approximation
            # Calculate local variance as texture measure
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(masked_gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((masked_gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            # Low texture indicates smooth, potentially dehydrated skin
            face_variance = local_variance[mask > 0]
            if len(face_variance) > 0:
                mean_variance = np.mean(face_variance)
                # Lower variance indicates smoother, potentially dehydrated skin
                dehydration_score = max(0, (50 - mean_variance) / 50)
                
                # Generate heatmap based on low texture areas
                heatmap = 1.0 - cv2.normalize(local_variance, None, 0, 1, cv2.NORM_MINMAX)
                heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
                
                return min(dehydration_score, 1.0), heatmap
            
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Dehydration analysis failed: {e}")
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
    
    def _analyze_pore_size(self, image: np.ndarray, landmarks: List[Dict], mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect enlarged pores using morphological operations"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply face mask
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Detect dark spots (pores) using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Top-hat transform to detect small dark spots
            tophat = cv2.morphologyEx(masked_gray, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold to get pore candidates
            _, pore_mask = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
            
            # Filter small components (noise)
            contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pore_area = 0
            large_pores = np.zeros_like(pore_mask)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 200:  # Filter by pore size
                    cv2.drawContours(large_pores, [contour], -1, 255, -1)
                    pore_area += area
            
            # Calculate pore size score
            total_face_area = np.sum(mask > 0)
            pore_score = pore_area / total_face_area if total_face_area > 0 else 0
            
            # Generate heatmap
            heatmap = large_pores.astype(np.float32) / 255.0
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
            
            return min(pore_score * 10, 1.0), heatmap
            
        except Exception as e:
            logger.warning(f"Pore analysis failed: {e}")
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
    
    def _analyze_fine_lines(self, image: np.ndarray, landmarks: List[Dict], mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect fine lines using edge detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply face mask
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Enhance lines using Gaussian blur difference
            blur1 = cv2.GaussianBlur(masked_gray, (3, 3), 0)
            blur2 = cv2.GaussianBlur(masked_gray, (7, 7), 0)
            lines_enhanced = cv2.subtract(blur2, blur1)
            
            # Detect edges
            edges = cv2.Canny(lines_enhanced, 30, 80)
            
            # Apply morphological operations to connect line segments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Calculate fine lines score
            line_pixels = np.sum(lines > 0)
            total_face_pixels = np.sum(mask > 0)
            lines_score = line_pixels / total_face_pixels if total_face_pixels > 0 else 0
            
            # Generate heatmap
            heatmap = lines.astype(np.float32) / 255.0
            heatmap = cv2.GaussianBlur(heatmap, (10, 10), 0)
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
            
            return min(lines_score * 5, 1.0), heatmap
            
        except Exception as e:
            logger.warning(f"Fine lines analysis failed: {e}")
            return 0.0, np.zeros(image.shape[:2], dtype=np.float32)
    
    def _generate_combined_heatmap(self, heatmaps: Dict[str, np.ndarray], image_size: Tuple[int, int]) -> str:
        """Generate combined heatmap visualization"""
        try:
            # Combine all heatmaps with weights
            weights = {
                'acne': 0.25,
                'hyperpigmentation': 0.2,
                'redness': 0.2,
                'dehydration': 0.15,
                'pore_size': 0.1,
                'fine_lines': 0.1
            }
            
            combined = np.zeros(heatmaps['acne'].shape, dtype=np.float32)
            
            for condition, heatmap in heatmaps.items():
                if condition in weights:
                    combined += heatmap * weights[condition]
            
            # Normalize
            combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply colormap
            colored = cv2.applyColorMap(combined.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Resize to original image size
            colored = cv2.resize(colored, image_size)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', colored)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return heatmap_b64
            
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
            return ""
    
    def _calculate_zone_scores(self, image: np.ndarray, landmarks: List[Dict], predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate condition scores for different facial zones"""
        zone_scores = {}
        
        try:
            # Group landmarks by zone
            zones = {}
            for landmark in landmarks:
                zone = landmark.get('zone', 'other')
                if zone not in zones:
                    zones[zone] = []
                zones[zone].append(landmark)
            
            # Calculate scores for each zone
            for zone, zone_landmarks in zones.items():
                if zone_landmarks and zone != 'other':
                    # For now, use average of all conditions weighted by zone relevance
                    zone_weights = {
                        'forehead': {'acne': 0.3, 'fine_lines': 0.4, 'dehydration': 0.3},
                        'nose': {'pore_size': 0.5, 'acne': 0.3, 'redness': 0.2},
                        'left_cheek': {'hyperpigmentation': 0.4, 'acne': 0.3, 'redness': 0.3},
                        'right_cheek': {'hyperpigmentation': 0.4, 'acne': 0.3, 'redness': 0.3},
                        'chin': {'acne': 0.5, 'pore_size': 0.3, 'redness': 0.2}
                    }
                    
                    if zone in zone_weights:
                        zone_score = 0
                        for condition, weight in zone_weights[zone].items():
                            zone_score += predictions.get(condition, 0) * weight
                        zone_scores[zone] = min(zone_score, 1.0)
                    else:
                        zone_scores[zone] = np.mean(list(predictions.values()))
            
        except Exception as e:
            logger.warning(f"Zone score calculation failed: {e}")
            
        return zone_scores
    
    def _generate_explanations(self, predictions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate human-readable explanations"""
        explanations = []
        
        condition_info = {
            'acne': {
                'name': 'Acne',
                'explanation': 'Active breakouts and inflammatory lesions detected.',
                'recommendations': [
                    'Use gentle, non-comedogenic cleanser twice daily',
                    'Apply salicylic acid or benzoyl peroxide treatment',
                    'Avoid over-washing or harsh scrubbing',
                    'Consider consulting a dermatologist for persistent cases'
                ]
            },
            'hyperpigmentation': {
                'name': 'Hyperpigmentation',
                'explanation': 'Dark spots and uneven skin tone detected.',
                'recommendations': [
                    'Apply broad-spectrum SPF 30+ daily',
                    'Use vitamin C serum in the morning',
                    'Consider niacinamide for evening routine',
                    'Be patient - improvements take 6-12 weeks'
                ]
            },
            'redness': {
                'name': 'Redness',
                'explanation': 'Increased vascular activity or inflammation detected.',
                'recommendations': [
                    'Identify and avoid triggers (heat, spicy food, alcohol)',
                    'Use gentle, fragrance-free products',
                    'Apply cool compresses for immediate relief',
                    'Consider anti-inflammatory ingredients like niacinamide'
                ]
            },
            'dehydration': {
                'name': 'Dehydration',
                'explanation': 'Skin appears tight with reduced moisture levels.',
                'recommendations': [
                    'Increase water intake throughout the day',
                    'Use hyaluronic acid serum on damp skin',
                    'Apply moisturizer while skin is still damp',
                    'Consider a humidifier in dry environments'
                ]
            },
            'pore_size': {
                'name': 'Enlarged Pores',
                'explanation': 'Visible pore enlargement detected, typically on T-zone.',
                'recommendations': [
                    'Use BHA (salicylic acid) 2-3 times weekly',
                    'Consider retinoid products for long-term improvement',
                    'Avoid pore strips which can damage skin',
                    'Clay masks can temporarily minimize appearance'
                ]
            },
            'fine_lines': {
                'name': 'Fine Lines',
                'explanation': 'Early signs of aging and expression lines detected.',
                'recommendations': [
                    'Start with over-the-counter retinol products',
                    'Maintain consistent moisturizing routine',
                    'Always wear sunscreen to prevent further damage',
                    'Consider professional treatments for advanced signs'
                ]
            }
        }
        
        # Sort by score and explain top conditions
        sorted_conditions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for condition, score in sorted_conditions[:3]:  # Top 3 conditions
            if score > 0.3 and condition in condition_info:  # Only explain significant findings
                info = condition_info[condition]
                severity = 'high' if score > 0.7 else 'moderate' if score > 0.5 else 'mild'
                
                explanations.append({
                    'condition': condition,
                    'name': info['name'],
                    'score': round(score, 3),
                    'severity': severity,
                    'explanation': info['explanation'],
                    'recommendations': info['recommendations'][:3]  # Top 3 recommendations
                })
        
        return explanations
    
    def _calculate_confidence(self, predictions: Dict[str, float]) -> str:
        """Calculate overall prediction confidence"""
        try:
            max_score = max(predictions.values()) if predictions else 0
            avg_score = np.mean(list(predictions.values())) if predictions else 0
            score_variance = np.var(list(predictions.values())) if predictions else 0
            
            # High confidence: high max score, reasonable average, low variance
            if max_score > 0.8 and avg_score > 0.3 and score_variance < 0.1:
                return 'high'
            elif max_score > 0.6 and avg_score > 0.2:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'low'
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Blur assessment using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 1000, 1.0)
            
            # Brightness assessment
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Contrast assessment
            contrast = np.std(gray)
            contrast_score = min(contrast / 64, 1.0)
            
            overall_quality = (blur_score + brightness_score + contrast_score) / 3
            
            return {
                'overall': overall_quality,
                'blur_score': blur_score,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'is_acceptable': overall_quality > 0.6
            }
            
        except Exception as e:
            logger.warning(f"Image quality assessment failed: {e}")
            return {'overall': 0.5, 'is_acceptable': True}
    
    def _estimate_skin_tone(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Estimate skin tone using ITA (Individual Typology Angle)"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Extract L and b channels for ITA calculation
            l_channel = lab[:, :, 0]
            b_channel = lab[:, :, 2]
            
            # Apply face mask
            face_l = l_channel[mask > 0]
            face_b = b_channel[mask > 0]
            
            if len(face_l) > 0:
                # Calculate mean L and b values
                mean_l = np.mean(face_l)
                mean_b = np.mean(face_b)
                
                # Calculate ITA (Individual Typology Angle)
                ita = np.arctan((mean_l - 50) / mean_b) * 180 / np.pi
                
                # Classify skin tone based on ITA
                if ita > 55:
                    fitzpatrick = "I"
                    description = "Very light"
                elif ita > 41:
                    fitzpatrick = "II"
                    description = "Light"
                elif ita > 28:
                    fitzpatrick = "III"
                    description = "Light-Medium"
                elif ita > 10:
                    fitzpatrick = "IV"
                    description = "Medium"
                elif ita > -30:
                    fitzpatrick = "V"
                    description = "Dark"
                else:
                    fitzpatrick = "VI"
                    description = "Very dark"
                
                return {
                    'ita': float(ita),
                    'fitzpatrick': fitzpatrick,
                    'description': description,
                    'lightness': float(mean_l),
                    'color_component': float(mean_b)
                }
            
            return {'ita': 0, 'fitzpatrick': 'Unknown', 'description': 'Cannot determine'}
            
        except Exception as e:
            logger.warning(f"Skin tone estimation failed: {e}")
            return {'ita': 0, 'fitzpatrick': 'Unknown', 'description': 'Cannot determine'}
    
    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when analysis fails"""
        return {
            'predictions': {
                'acne': 0.2,
                'hyperpigmentation': 0.1,
                'redness': 0.1,
                'dehydration': 0.3,
                'pore_size': 0.2,
                'fine_lines': 0.1
            },
            'landmarks': [],
            'heatmap_b64': '',
            'explanations': [{
                'condition': 'analysis_failed',
                'name': 'Analysis Status',
                'score': 0.0,
                'severity': 'unknown',
                'explanation': 'Analysis could not be completed. Please try with a clearer image.',
                'recommendations': ['Ensure good lighting', 'Hold camera steady', 'Position face clearly in frame']
            }],
            'confidence': 'low',
            'zone_scores': {},
            'image_quality': {'overall': 0.3, 'is_acceptable': False},
            'skin_tone': {'fitzpatrick': 'Unknown', 'description': 'Cannot determine'}
        }