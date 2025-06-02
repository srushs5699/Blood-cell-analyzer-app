import cv2
import numpy as np
from typing import Dict, List, Tuple

class ImageProcessor:
    def __init__(self):
        self.colors = {
            'RBC': (255, 0, 0),      # Red
            'WBC': (0, 255, 0),      # Green
            'Platelet': (0, 0, 255)  # Blue
        }
    
    def process_detection_results(self, results: Dict, original_image: np.ndarray) -> Dict:
        """
        Process YOLO detection results and create annotated image
        
        Args:
            results: Detection results from YOLO model
            original_image: Original input image
            
        Returns:
            Dictionary containing processed results
        """
        detections = results.get('detections', [])
        
        # Count cells by type
        cell_counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0}
        confidences = []
        
        # Create annotated image
        annotated_image = original_image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Count cells
            if class_name in cell_counts:
                cell_counts[class_name] += 1
            
            confidences.append(confidence)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'rbc_count': cell_counts['RBC'],
            'wbc_count': cell_counts['WBC'],
            'platelet_count': cell_counts['Platelet'],
            'avg_confidence': float(avg_confidence),
            'annotated_image': annotated_image,
            'detections': detections
        }
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better detection
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        return enhanced