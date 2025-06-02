import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.yolo_model import BloodCellYOLO
    from models.cell_classifier import CellClassifier
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Models not available: {e}")
    MODELS_AVAILABLE = False

@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestYOLOModel:
    """Test YOLO model functionality"""
    
    def test_model_initialization(self):
        """Test YOLO model can be initialized"""
        model = BloodCellYOLO()
        assert model is not None
        assert hasattr(model, 'class_names')
        assert len(model.class_names) == 3

    def test_model_detection(self):
        """Test YOLO model detection"""
        model = BloodCellYOLO()
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        results = model.detect(test_image)
        assert 'detections' in results
        assert 'image_shape' in results
        assert isinstance(results['detections'], list)

@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestCellClassifier:
    """Test cell classifier functionality"""
    
    def test_classifier_initialization(self):
        """Test classifier can be initialized"""
        classifier = CellClassifier()
        assert classifier is not None
        assert hasattr(classifier, 'class_names')

    def test_cell_classification(self):
        """Test cell classification"""
        classifier = CellClassifier()
        
        # Create test cell crop
        cell_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        result = classifier.classify_cell_crop(cell_crop)
        assert 'predicted_class' in result
        assert 'class_name' in result
        assert 'confidence' in result
        assert result['class_name'] in ['RBC', 'WBC', 'Platelet']
