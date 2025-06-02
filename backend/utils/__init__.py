# backend/utils/__init__.py
"""
Blood Cell Analyzer Utilities Package

This package contains utility modules for image processing, Firebase integration,
data validation, and other helper functions for the blood cell analyzer.

Modules:
    - image_processor: Image processing and enhancement utilities
    - firebase_config: Firebase integration and data management
    - validators: Input validation and data sanitization
    - metrics: Performance metrics and analytics
    - file_handler: File upload and management utilities
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Configure logging for the utils package
logger = logging.getLogger(__name__)

# Version and package info
__version__ = "1.0.0"
__author__ = "Blood Cell Analyzer Team"

# Import utility classes and functions
try:
    from .image_processor import ImageProcessor
    from .firebase_config import FirebaseService, initialize_firebase
    
    # Mark successful imports
    IMAGE_PROCESSOR_AVAILABLE = True
    FIREBASE_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Some utilities failed to import: {e}")
    ImageProcessor = None
    FirebaseService = None
    initialize_firebase = None
    
    IMAGE_PROCESSOR_AVAILABLE = False
    FIREBASE_AVAILABLE = False

# Additional utility imports
try:
    from .validators import (
        ImageValidator, 
        DataValidator, 
        validate_analysis_request,
        sanitize_filename
    )
    VALIDATORS_AVAILABLE = True
except ImportError:
    ImageValidator = None
    DataValidator = None
    validate_analysis_request = None
    sanitize_filename = None
    VALIDATORS_AVAILABLE = False

try:
    from .metrics import (
        PerformanceMetrics,
        AnalyticsCollector,
        calculate_cell_statistics
    )
    METRICS_AVAILABLE = True
except ImportError:
    PerformanceMetrics = None
    AnalyticsCollector = None
    calculate_cell_statistics = None
    METRICS_AVAILABLE = False

try:
    from .file_handler import (
        FileManager,
        SecureFileUpload,
        generate_unique_filename
    )
    FILE_HANDLER_AVAILABLE = True
except ImportError:
    FileManager = None
    SecureFileUpload = None
    generate_unique_filename = None
    FILE_HANDLER_AVAILABLE = False

# Export all available utilities
__all__ = [
    # Core utilities
    'ImageProcessor',
    'FirebaseService', 
    'initialize_firebase',
    
    # Validation utilities
    'ImageValidator',
    'DataValidator',
    'validate_analysis_request',
    'sanitize_filename',
    
    # Metrics and analytics
    'PerformanceMetrics',
    'AnalyticsCollector', 
    'calculate_cell_statistics',
    
    # File handling
    'FileManager',
    'SecureFileUpload',
    'generate_unique_filename',
    
    # Utility manager
    'UtilityManager',
    'get_utils_status',
    'setup_directories',
    
    # Availability flags
    'IMAGE_PROCESSOR_AVAILABLE',
    'FIREBASE_AVAILABLE',
    'VALIDATORS_AVAILABLE',
    'METRICS_AVAILABLE',
    'FILE_HANDLER_AVAILABLE'
]

class UtilityManager:
    """
    Central manager for all utility functions and services
    Provides a unified interface for accessing all utilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the utility manager
        
        Args:
            config: Configuration dictionary for utilities
        """
        self.config = config or {}
        self._initialize_utilities()
        
        logger.info("UtilityManager initialized successfully")
    
    def _initialize_utilities(self):
        """Initialize all available utilities"""
        
        # Image processor
        if IMAGE_PROCESSOR_AVAILABLE:
            try:
                self.image_processor = ImageProcessor()
                logger.info("Image processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize image processor: {e}")
                self.image_processor = None
        else:
            self.image_processor = None
        
        # Firebase service
        if FIREBASE_AVAILABLE:
            try:
                self.firebase_service = FirebaseService()
                logger.info("Firebase service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase service: {e}")
                self.firebase_service = None
        else:
            self.firebase_service = None
        
        # Validators
        if VALIDATORS_AVAILABLE:
            try:
                self.image_validator = ImageValidator()
                self.data_validator = DataValidator()
                logger.info("Validators initialized")
            except Exception as e:
                logger.error(f"Failed to initialize validators: {e}")
                self.image_validator = None
                self.data_validator = None
        else:
            self.image_validator = None
            self.data_validator = None
        
        # Metrics collector
        if METRICS_AVAILABLE:
            try:
                self.metrics_collector = AnalyticsCollector()
                self.performance_metrics = PerformanceMetrics()
                logger.info("Metrics services initialized")
            except Exception as e:
                logger.error(f"Failed to initialize metrics: {e}")
                self.metrics_collector = None
                self.performance_metrics = None
        else:
            self.metrics_collector = None
            self.performance_metrics = None
        
        # File manager
        if FILE_HANDLER_AVAILABLE:
            try:
                upload_dir = self.config.get('upload_directory', 'uploads')
                self.file_manager = FileManager(upload_dir)
                self.secure_upload = SecureFileUpload()
                logger.info("File handling services initialized")
            except Exception as e:
                logger.error(f"Failed to initialize file handlers: {e}")
                self.file_manager = None
                self.secure_upload = None
        else:
            self.file_manager = None
            self.secure_upload = None
    
    def process_image(self, image, enhance: bool = True):
        """
        Process an image using the image processor
        
        Args:
            image: Input image
            enhance: Whether to apply enhancement
            
        Returns:
            Processed image or None if processor unavailable
        """
        if self.image_processor:
            if enhance:
                return self.image_processor.enhance_image_quality(image)
            return image
        else:
            logger.warning("Image processor not available")
            return image
    
    def validate_upload(self, file_data, filename: str) -> Dict[str, Any]:
        """
        Validate an uploaded file
        
        Args:
            file_data: File data to validate
            filename: Original filename
            
        Returns:
            Validation result
        """
        if self.image_validator:
            return self.image_validator.validate_file(file_data, filename)
        else:
            return {
                'valid': True,
                'message': 'Validation skipped - validator not available',
                'warnings': ['Image validation not available']
            }
    
    def save_analysis_result(self, analysis_data: Dict[str, Any]) -> Optional[str]:
        """
        Save analysis results to Firebase
        
        Args:
            analysis_data: Analysis results to save
            
        Returns:
            Document ID if successful, None otherwise
        """
        if self.firebase_service:
            return self.firebase_service.save_analysis_result(analysis_data)
        else:
            logger.warning("Firebase service not available for saving results")
            return None
    
    def collect_metrics(self, analysis_data: Dict[str, Any]):
        """
        Collect performance and analytics metrics
        
        Args:
            analysis_data: Analysis data for metrics collection
        """
        if self.metrics_collector:
            self.metrics_collector.record_analysis(analysis_data)
        
        if self.performance_metrics:
            self.performance_metrics.record_processing_time(
                analysis_data.get('processing_time', 0)
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all utility services
        
        Returns:
            Status dictionary
        """
        return {
            'image_processor': self.image_processor is not None,
            'firebase_service': self.firebase_service is not None,
            'image_validator': self.image_validator is not None,
            'data_validator': self.data_validator is not None,
            'metrics_collector': self.metrics_collector is not None,
            'performance_metrics': self.performance_metrics is not None,
            'file_manager': self.file_manager is not None,
            'secure_upload': self.secure_upload is not None
        }

def get_utils_status() -> Dict[str, bool]:
    """
    Get availability status of all utility modules
    
    Returns:
        Dictionary indicating which utilities are available
    """
    return {
        'image_processor': IMAGE_PROCESSOR_AVAILABLE,
        'firebase': FIREBASE_AVAILABLE,
        'validators': VALIDATORS_AVAILABLE,
        'metrics': METRICS_AVAILABLE,
        'file_handler': FILE_HANDLER_AVAILABLE
    }

def setup_directories(base_path: str = ".") -> Dict[str, str]:
    """
    Setup required directories for the application
    
    Args:
        base_path: Base path for directory creation
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        'uploads': 'uploads',
        'results': 'results',
        'models': 'models',
        'logs': 'logs',
        'temp': 'temp'
    }
    
    created_dirs = {}
    
    for name, path in directories.items():
        full_path = os.path.join(base_path, path)
        try:
            os.makedirs(full_path, exist_ok=True)
            created_dirs[name] = full_path
            logger.info(f"Directory ensured: {full_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {full_path}: {e}")
            created_dirs[name] = None
    
    return created_dirs

def cleanup_temp_files(temp_dir: str = "temp", max_age_hours: int = 24):
    """
    Cleanup temporary files older than specified age
    
    Args:
        temp_dir: Temporary directory to clean
        max_age_hours: Maximum age in hours before deletion
    """
    try:
        import time
        
        if not os.path.exists(temp_dir):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        deleted_count = 0
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old temp file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} temporary files")
    
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for diagnostics
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    try:
        return {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percentage': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free
            },
            'cpu': {
                'count': psutil.cpu_count(),
                'usage': psutil.cpu_percent(interval=1)
            }
        }
    except ImportError:
        logger.warning("psutil not available for system info")
        return {
            'platform': {
                'system': platform.system(),
                'release': platform.release()
            },
            'memory': {'status': 'unavailable'},
            'disk': {'status': 'unavailable'},
            'cpu': {'status': 'unavailable'}
        }

def validate_environment() -> Dict[str, Any]:
    """
    Validate that the environment is properly configured
    
    Returns:
        Validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check required directories
    required_dirs = ['uploads', 'models', 'logs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            validation_results['warnings'].append(f"Directory {dir_name} does not exist")
    
    # Check utility availability
    utils_status = get_utils_status()
    unavailable_utils = [name for name, available in utils_status.items() if not available]
    
    if unavailable_utils:
        validation_results['warnings'].extend([
            f"Utility {util} is not available" for util in unavailable_utils
        ])
    
    # Check environment variables
    important_env_vars = ['SECRET_KEY', 'FLASK_ENV']
    for var in important_env_vars:
        if not os.getenv(var):
            validation_results['warnings'].append(f"Environment variable {var} not set")
    
    # Add info about available utilities
    available_utils = [name for name, available in utils_status.items() if available]
    validation_results['info'] = [
        f"Available utilities: {', '.join(available_utils)}"
    ]
    
    return validation_results

# Package initialization
def initialize_utils_package():
    """Initialize the utils package"""
    logger.info(f"Blood Cell Analyzer Utils Package v{__version__}")
    
    # Setup basic directories
    setup_directories()
    
    # Validate environment
    validation = validate_environment()
    
    if validation['errors']:
        logger.error("Critical environment issues found:")
        for error in validation['errors']:
            logger.error(f"  - {error}")
    
    if validation['warnings']:
        logger.warning("Environment warnings:")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
    
    # Log available utilities
    status = get_utils_status()
    available = [name for name, avail in status.items() if avail]
    unavailable = [name for name, avail in status.items() if not avail]
    
    if available:
        logger.info(f"Available utilities: {', '.join(available)}")
    if unavailable:
        logger.info(f"Unavailable utilities: {', '.join(unavailable)}")

# Auto-initialize when package is imported
try:
    initialize_utils_package()
except Exception as e:
    logger.error(f"Failed to initialize utils package: {e}")

# Convenience function for quick utility manager setup
def create_utility_manager(config: Optional[Dict[str, Any]] = None) -> UtilityManager:
    """
    Create a UtilityManager instance with optional configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UtilityManager instance
    """
    return UtilityManager(config)