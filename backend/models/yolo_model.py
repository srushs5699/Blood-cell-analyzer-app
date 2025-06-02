import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Dict, Tuple
import yaml

class BloodCellYOLO:
    def __init__(self, model_path='models/yolov5_blood_cells.pt', confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.class_names = ['RBC', 'WBC', 'Platelet']
        
        # Load YOLOv5 model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
        except:
            # Fallback: Create a mock model for demonstration
            self.model = self._create_mock_model()
            print("Warning: Using mock model for demonstration. Replace with actual trained YOLOv5 model.")
    
    def _create_mock_model(self):
        """Create a mock model that simulates blood cell detection for demonstration"""
        class MockYOLOModel:
            def __call__(self, image):
                # Simulate detection results
                height, width = image.shape[:2]
                
                # Generate random detections for demonstration
                np.random.seed(42)  # For consistent results
                num_detections = np.random.randint(15, 35)
                
                detections = []
                for _ in range(num_detections):
                    x1 = np.random.randint(0, width//2)
                    y1 = np.random.randint(0, height//2)
                    x2 = x1 + np.random.randint(20, 60)
                    y2 = y1 + np.random.randint(20, 60)
                    
                    # Ensure coordinates are within image bounds
                    x2 = min(x2, width)
                    y2 = min(y2, height)
                    
                    class_id = np.random.randint(0, 3)  # 0: RBC, 1: WBC, 2: Platelet
                    confidence = 0.7 + np.random.random() * 0.25  # 0.7-0.95 confidence
                    
                    detections.append([x1, y1, x2, y2, confidence, class_id])
                
                return MockResults(detections)
        
        class MockResults:
            def __init__(self, detections):
                self.xyxy = [torch.tensor(detections)]
        
        return MockYOLOModel()
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect blood cells in the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Preprocess image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run inference
            results = self.model(image_rgb)
            
            # Process results
            detections = []
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                for detection in results.xyxy[0]:  # results.xyxy[0] contains detections for first image
                    x1, y1, x2, y2, conf, cls = detection.tolist()
                    
                    if conf >= self.confidence_threshold:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': int(cls),
                            'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else 'Unknown'
                        })
            
            return {
                'detections': detections,
                'image_shape': image.shape
            }
            
        except Exception as e:
            print(f"Detection error: {e}")
            return {'detections': [], 'image_shape': image.shape}
