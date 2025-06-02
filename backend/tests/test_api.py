import pytest
import json
import io
import os
import sys
from PIL import Image
import numpy as np

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a 224x224 RGB image with random data
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to BytesIO object
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    
    return img_io

class TestHealthEndpoint:
    """Test the health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'message' in data

class TestAnalysisEndpoints:
    """Test analysis endpoints"""
    
    def test_analyze_no_image(self, client):
        """Test analyze endpoint without image"""
        response = client.post('/api/analyze')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data

    def test_analyze_empty_filename(self, client):
        """Test analyze endpoint with empty filename"""
        response = client.post('/api/analyze', 
                              data={'image': (io.BytesIO(), '')})
        assert response.status_code == 400

    def test_analyze_with_valid_image(self, client, sample_image):
        """Test analyze endpoint with valid image"""
        response = client.post('/api/analyze',
                              data={'image': (sample_image, 'test.jpg')},
                              content_type='multipart/form-data')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'cell_counts' in data
        assert 'processing_time' in data
        assert 'confidence_score' in data
        assert 'analysis_id' in data
        
        # Check cell counts structure
        cell_counts = data['cell_counts']
        assert 'RBC' in cell_counts
        assert 'WBC' in cell_counts
        assert 'Platelets' in cell_counts
        
        # Check data types
        assert isinstance(data['processing_time'], (int, float))
        assert isinstance(data['confidence_score'], (int, float))
        assert isinstance(data['total_cells_detected'], int)

    def test_analyze_processing_time(self, client, sample_image):
        """Test that processing time is reasonable"""
        response = client.post('/api/analyze',
                              data={'image': (sample_image, 'test.jpg')},
                              content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Processing should be under 10 seconds (generous for testing)
        assert data['processing_time'] < 10.0

class TestModelIntegration:
    """Test model integration and responses"""
    
    def test_model_confidence_scores(self, client, sample_image):
        """Test that confidence scores are in valid range"""
        response = client.post('/api/analyze',
                              data={'image': (sample_image, 'test.jpg')},
                              content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Confidence should be between 0 and 1
        assert 0 <= data['confidence_score'] <= 1

    def test_cell_count_consistency(self, client, sample_image):
        """Test that cell counts are consistent"""
        response = client.post('/api/analyze',
                              data={'image': (sample_image, 'test.jpg')},
                              content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        cell_counts = data['cell_counts']
        total_from_counts = sum(cell_counts.values())
        total_detected = data['total_cells_detected']
        
        # Totals should match
        assert total_from_counts == total_detected
