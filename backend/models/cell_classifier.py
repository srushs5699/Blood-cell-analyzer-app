import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CellClassifierCNN(nn.Module):
    """
    Convolutional Neural Network for blood cell classification
    Architecture optimized for RBC, WBC, and Platelet classification
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(CellClassifierCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CellClassifier:
    """
    Blood cell classifier for refining YOLO detection results
    Provides secondary validation and detailed cell analysis
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the cell classifier
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.class_names = ['RBC', 'WBC', 'Platelet']
        self.num_classes = len(self.class_names)
        self.input_size = (64, 64)  # Standard input size for cell crops
        
        # Initialize model
        self.model = CellClassifierCNN(num_classes=self.num_classes)
        self.model.to(self.device)
        
        # Load pre-trained weights if available
        if model_path and self._load_model(model_path):
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.warning("Using randomly initialized model - train or provide pre-trained weights for better performance")
        
        self.model.eval()
        
        # Define image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Cell morphology features for additional analysis
        self.morphology_analyzer = CellMorphologyAnalyzer()
    
    def _load_model(self, model_path: str) -> bool:
        """
        Load pre-trained model weights
        
        Args:
            model_path: Path to model weights file
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def preprocess_cell_image(self, cell_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess cell image for classification
        
        Args:
            cell_image: Input cell image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to PIL Image if numpy array
        if isinstance(cell_image, np.ndarray):
            # Ensure image is in RGB format
            if len(cell_image.shape) == 3 and cell_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if cell_image.dtype == np.uint8:
                    cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
            elif len(cell_image.shape) == 2:
                # Convert grayscale to RGB
                cell_image = cv2.cvtColor(cell_image, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            cell_image = Image.fromarray(cell_image.astype(np.uint8))
        
        # Apply transformations
        image_tensor = self.transform(cell_image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def classify_cell_crop(self, cell_image: np.ndarray, return_features: bool = False) -> Dict:
        """
        Classify a cropped cell image
        
        Args:
            cell_image: Cropped image of a single cell
            return_features: Whether to return additional morphological features
            
        Returns:
            Dict: Classification results with confidence scores and optional features
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_cell_image(cell_image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get class probabilities
            class_probs = {
                self.class_names[i]: probabilities[0][i].item() 
                for i in range(self.num_classes)
            }
            
            result = {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'class_probabilities': class_probs,
                'processing_successful': True
            }
            
            # Add morphological features if requested
            if return_features:
                morphology_features = self.morphology_analyzer.analyze(cell_image)
                result['morphology_features'] = morphology_features
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                'predicted_class': 0,
                'class_name': 'RBC',
                'confidence': 0.33,
                'class_probabilities': {name: 0.33 for name in self.class_names},
                'processing_successful': False,
                'error': str(e)
            }
    
    def classify_batch(self, cell_images: List[np.ndarray], batch_size: int = 32) -> List[Dict]:
        """
        Classify multiple cell images in batches
        
        Args:
            cell_images: List of cell image arrays
            batch_size: Number of images to process per batch
            
        Returns:
            List[Dict]: Classification results for each image
        """
        results = []
        
        for i in range(0, len(cell_images), batch_size):
            batch_images = cell_images[i:i + batch_size]
            batch_results = []
            
            try:
                # Preprocess batch
                batch_tensors = []
                for img in batch_images:
                    tensor = self.preprocess_cell_image(img)
                    batch_tensors.append(tensor.squeeze(0))
                
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors)
                    
                    # Run batch inference
                    with torch.no_grad():
                        outputs = self.model(batch_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        predicted_classes = torch.argmax(probabilities, dim=1)
                    
                    # Process results
                    for j, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
                        pred_class_idx = pred_class.item()
                        confidence = probs[pred_class_idx].item()
                        
                        class_probs = {
                            self.class_names[k]: probs[k].item() 
                            for k in range(self.num_classes)
                        }
                        
                        batch_results.append({
                            'predicted_class': pred_class_idx,
                            'class_name': self.class_names[pred_class_idx],
                            'confidence': confidence,
                            'class_probabilities': class_probs,
                            'processing_successful': True
                        })
                
            except Exception as e:
                logger.error(f"Batch classification error: {e}")
                # Add error results for this batch
                for _ in batch_images:
                    batch_results.append({
                        'predicted_class': 0,
                        'class_name': 'RBC',
                        'confidence': 0.33,
                        'class_probabilities': {name: 0.33 for name in self.class_names},
                        'processing_successful': False,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
        
        return results
    
    def get_cell_features(self, cell_image: np.ndarray) -> Dict:
        """
        Extract detailed features from a cell image
        
        Args:
            cell_image: Cell image array
            
        Returns:
            Dict: Detailed cell features
        """
        classification = self.classify_cell_crop(cell_image, return_features=True)
        
        # Add size and shape analysis
        features = {
            'classification': classification,
            'image_properties': {
                'width': cell_image.shape[1],
                'height': cell_image.shape[0],
                'area_pixels': cell_image.shape[0] * cell_image.shape[1],
                'channels': cell_image.shape[2] if len(cell_image.shape) > 2 else 1
            }
        }
        
        return features

class CellMorphologyAnalyzer:
    """
    Analyzer for extracting morphological features from cell images
    """
    
    def analyze(self, cell_image: np.ndarray) -> Dict:
        """
        Analyze morphological features of a cell
        
        Args:
            cell_image: Cell image array
            
        Returns:
            Dict: Morphological features
        """
        try:
            # Convert to grayscale for analysis
            if len(cell_image.shape) == 3:
                gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = cell_image.copy()
            
            # Basic shape analysis
            contours, _ = cv2.findContours(
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Find largest contour (main cell body)
                main_contour = max(contours, key=cv2.contourArea)
                
                # Calculate morphological features
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(main_contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Convex hull features
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Equivalent diameter
                equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
                
                return {
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'aspect_ratio': float(aspect_ratio),
                    'circularity': float(circularity),
                    'solidity': float(solidity),
                    'equivalent_diameter': float(equiv_diameter),
                    'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'contour_points': len(main_contour)
                }
            else:
                return self._default_morphology_features()
                
        except Exception as e:
            logger.error(f"Morphology analysis error: {e}")
            return self._default_morphology_features()
    
    def _default_morphology_features(self) -> Dict:
        """Return default morphology features when analysis fails"""
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'aspect_ratio': 1.0,
            'circularity': 0.0,
            'solidity': 0.0,
            'equivalent_diameter': 0.0,
            'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
            'contour_points': 0
        }

class CellClassifierTrainer:
    """
    Trainer class for training the cell classifier model
    """
    
    def __init__(self, model: CellClassifierCNN, device: str = None):
        self.model = model
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """Setup training components"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, filepath: str, epoch: int, best_acc: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': best_acc,
        }, filepath)

# Example usage and testing functions
def test_classifier():
    """Test the cell classifier with sample data"""
    print("Testing Cell Classifier...")
    
    # Initialize classifier
    classifier = CellClassifier()
    
    # Create sample cell image
    sample_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Test single classification
    result = classifier.classify_cell_crop(sample_image, return_features=True)
    print(f"Classification result: {result}")
    
    # Test batch classification
    batch_images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
    batch_results = classifier.classify_batch(batch_images)
    print(f"Batch classification completed: {len(batch_results)} results")
    
    print("Cell Classifier test completed successfully!")

if __name__ == "__main__":
    test_classifier()