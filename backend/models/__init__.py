# backend/models/__init__.py
"""
Blood Cell Analyzer Models Package

This package contains all machine learning models and related utilities
for blood cell detection and classification.

Modules:
    - yolo_model: YOLOv5-based blood cell detection
    - cell_classifier: CNN-based cell classification and morphological analysis
"""

import logging
import torch
from typing import Optional, Dict, Any

# Configure logging for the models package
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "Blood Cell Analyzer Team"

# Import main model classes
try:
    from .yolo_model import BloodCellYOLO
    from .cell_classifier import CellClassifier, CellClassifierCNN, CellMorphologyAnalyzer
    
    # Model availability flags
    YOLO_AVAILABLE = True
    CLASSIFIER_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Failed to import some models: {e}")
    BloodCellYOLO = None
    CellClassifier = None
    CellClassifierCNN = None
    CellMorphologyAnalyzer = None
    
    YOLO_AVAILABLE = False
    CLASSIFIER_AVAILABLE = False

# Export main classes
__all__ = [
    'BloodCellYOLO',
    'CellClassifier', 
    'CellClassifierCNN',
    'CellMorphologyAnalyzer',
    'ModelManager',
    'get_device_info',
    'check_model_availability',
    'YOLO_AVAILABLE',
    'CLASSIFIER_AVAILABLE'
]

class ModelManager:
    """
    Central manager for all blood cell analysis models
    Handles model initialization, device management, and coordination
    """
    
    def __init__(self, 
                 yolo_model_path: Optional[str] = None,
                 classifier_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the model manager
        
        Args:
            yolo_model_path: Path to YOLOv5 model weights
            classifier_model_path: Path to classifier model weights
            device: Device to run models on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.yolo_model = None
        self.classifier_model = None
        
        # Initialize models
        self._initialize_models(yolo_model_path, classifier_model_path)
        
        logger.info(f"ModelManager initialized on device: {self.device}")
    
    def _initialize_models(self, yolo_path: Optional[str], classifier_path: Optional[str]):
        """Initialize available models"""
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = BloodCellYOLO(model_path=yolo_path)
                logger.info("YOLO model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO model: {e}")
                self.yolo_model = None
        
        # Initialize classifier model
        if CLASSIFIER_AVAILABLE:
            try:
                self.classifier_model = CellClassifier(model_path=classifier_path, device=str(self.device))
                logger.info("Classifier model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize classifier model: {e}")
                self.classifier_model = None
    
    def detect_and_classify(self, image, use_classifier: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: detect cells with YOLO and classify with CNN
        
        Args:
            image: Input image as numpy array
            use_classifier: Whether to use secondary classifier for refinement
            
        Returns:
            Dict containing detection and classification results
        """
        results = {
            'yolo_results': None,
            'classifier_results': None,
            'combined_results': None,
            'processing_info': {
                'yolo_used': False,
                'classifier_used': False,
                'device': str(self.device)
            }
        }
        
        # Step 1: YOLO Detection
        if self.yolo_model is not None:
            try:
                yolo_results = self.yolo_model.detect(image)
                results['yolo_results'] = yolo_results
                results['processing_info']['yolo_used'] = True
                logger.debug(f"YOLO detected {len(yolo_results.get('detections', []))} objects")
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
                return results
        else:
            logger.warning("YOLO model not available")
            return results
        
        # Step 2: Secondary Classification (if requested and available)
        if use_classifier and self.classifier_model is not None and yolo_results.get('detections'):
            try:
                classifier_results = self._refine_with_classifier(image, yolo_results['detections'])
                results['classifier_results'] = classifier_results
                results['processing_info']['classifier_used'] = True
                logger.debug("Classifier refinement completed")
            except Exception as e:
                logger.error(f"Classifier refinement failed: {e}")
        
        # Step 3: Combine results
        results['combined_results'] = self._combine_results(
            results['yolo_results'], 
            results['classifier_results']
        )
        
        return results
    
    def _refine_with_classifier(self, image, detections):
        """Use classifier to refine YOLO detections"""
        refined_detections = []
        
        for detection in detections:
            try:
                # Extract cell crop from image
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # Valid crop
                    cell_crop = image[y1:y2, x1:x2]
                    
                    # Classify the crop
                    classification = self.classifier_model.classify_cell_crop(cell_crop, return_features=True)
                    
                    # Combine YOLO and classifier results
                    refined_detection = {
                        **detection,  # Original YOLO detection
                        'refined_classification': classification,
                        'classifier_confidence': classification.get('confidence', 0),
                        'morphology_features': classification.get('morphology_features', {})
                    }
                    
                    refined_detections.append(refined_detection)
                
            except Exception as e:
                logger.warning(f"Failed to refine detection: {e}")
                # Keep original detection if refinement fails
                refined_detections.append(detection)
        
        return refined_detections
    
    def _combine_results(self, yolo_results, classifier_results):
        """Combine YOLO and classifier results into final output"""
        if not yolo_results:
            return None
        
        combined = {
            'detections': classifier_results if classifier_results else yolo_results.get('detections', []),
            'cell_counts': {},
            'confidence_metrics': {},
            'image_shape': yolo_results.get('image_shape')
        }
        
        # Count cells by type
        detections = combined['detections']
        for detection in detections:
            # Use refined classification if available, otherwise use YOLO classification
            if 'refined_classification' in detection:
                class_name = detection['refined_classification']['class_name']
            else:
                class_name = detection.get('class_name', 'Unknown')
            
            combined['cell_counts'][class_name] = combined['cell_counts'].get(class_name, 0) + 1
        
        # Calculate confidence metrics
        if detections:
            yolo_confidences = [d.get('confidence', 0) for d in detections]
            classifier_confidences = [
                d.get('classifier_confidence', d.get('confidence', 0)) 
                for d in detections
            ]
            
            combined['confidence_metrics'] = {
                'avg_yolo_confidence': sum(yolo_confidences) / len(yolo_confidences),
                'avg_classifier_confidence': sum(classifier_confidences) / len(classifier_confidences),
                'total_detections': len(detections)
            }
        
        return combined
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'yolo_available': self.yolo_model is not None,
            'classifier_available': self.classifier_model is not None,
            'device': str(self.device),
            'yolo_classes': self.yolo_model.class_names if self.yolo_model else None,
            'classifier_classes': self.classifier_model.class_names if self.classifier_model else None
        }
    
    def update_model_paths(self, yolo_path: Optional[str] = None, classifier_path: Optional[str] = None):
        """Update model paths and reload models"""
        if yolo_path or classifier_path:
            self._initialize_models(yolo_path, classifier_path)

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices
    
    Returns:
        Dict containing device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info['cuda_version'] = torch.version.cuda
        device_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        device_info['gpu_memory'] = [
            {
                'device': i,
                'total_memory': torch.cuda.get_device_properties(i).total_memory,
                'allocated': torch.cuda.memory_allocated(i),
                'cached': torch.cuda.memory_reserved(i)
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return device_info

def check_model_availability() -> Dict[str, bool]:
    """
    Check which models are available for use
    
    Returns:
        Dict indicating model availability
    """
    return {
        'yolo_model': YOLO_AVAILABLE,
        'cell_classifier': CLASSIFIER_AVAILABLE,
        'torch_available': True,  # If we got here, torch is available
        'device_cuda': torch.cuda.is_available()
    }

# Package initialization
def initialize_package():
    """Initialize the models package"""
    logger.info(f"Blood Cell Analyzer Models Package v{__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    device_info = get_device_info()
    logger.info(f"Running on: {device_info['current_device']}")
    
    if device_info['cuda_available']:
        logger.info(f"CUDA devices available: {device_info['cuda_device_count']}")
    
    availability = check_model_availability()
    logger.info(f"Model availability: {availability}")

# Auto-initialize when package is imported
initialize_package()

# Convenience function for quick model setup
def create_model_manager(yolo_path: Optional[str] = None, 
                        classifier_path: Optional[str] = None,
                        device: Optional[str] = None) -> ModelManager:
    """
    Convenience function to create a ModelManager instance
    
    Args:
        yolo_path: Path to YOLOv5 model
        classifier_path: Path to classifier model
        device: Device to use
        
    Returns:
        ModelManager instance
    """
    return ModelManager(yolo_path, classifier_path, device)