# backend/utils/config.py
"""
Configuration Management for Blood Cell Analyzer

This module provides comprehensive configuration management including:
- Environment variable handling
- Configuration validation
- Default values and fallbacks
- Dynamic configuration updates
- Configuration profiles for different environments
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import timedelta
import warnings

# Third-party imports with fallbacks
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    warnings.warn("python-dotenv not available - environment loading will be limited")

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    yolo_model_path: str = "models/yolov5_blood_cells.pt"
    classifier_model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 1000
    input_size: int = 640
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 1
    use_fp16: bool = False
    model_cache_size: int = 2  # Number of models to keep in memory

@dataclass
class FileConfig:
    """Configuration for file handling"""
    upload_folder: str = "uploads"
    results_folder: str = "results"
    temp_folder: str = "temp"
    logs_folder: str = "logs"
    models_folder: str = "models"
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'tiff', 'bmp'])
    cleanup_temp_files: bool = True
    temp_file_max_age_hours: int = 24
    max_concurrent_uploads: int = 10

@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    # SQLite (local database)
    sqlite_path: str = "data/blood_analyzer.db"
    sqlite_timeout: int = 30
    
    # Firebase
    firebase_project_id: Optional[str] = None
    firebase_credentials_path: Optional[str] = None
    firebase_storage_bucket: Optional[str] = None
    firebase_database_url: Optional[str] = None
    
    # Connection settings
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class APIConfig:
    """Configuration for API settings"""
    host: str = "localhost"
    port: int = 5000
    debug: bool = False
    secret_key: str = "dev-secret-key-change-in-production"
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    rate_limit_per_minute: int = 60
    request_timeout: int = 300  # 5 minutes
    enable_swagger: bool = True
    api_version: str = "v1"

@dataclass
class SecurityConfig:
    """Security-related configuration"""
    jwt_secret_key: Optional[str] = None
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    max_login_attempts: int = 5
    account_lockout_minutes: int = 15
    enable_csrf_protection: bool = True
    secure_cookies: bool = False  # Set to True in production with HTTPS
    session_timeout_minutes: int = 60
    api_key_required: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_file: str = "logs/blood_analyzer.log"
    access_log_file: str = "logs/access.log"
    error_log_file: str = "logs/error.log"

@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    enable_caching: bool = True
    cache_timeout_seconds: int = 300
    max_cache_size: int = 1000
    enable_compression: bool = True
    worker_processes: int = 1
    worker_threads: int = 4
    max_queue_size: int = 100
    enable_profiling: bool = False
    memory_limit_mb: int = 2048

@dataclass
class AnalysisConfig:
    """Analysis-specific configuration"""
    default_cell_types: List[str] = field(default_factory=lambda: ['RBC', 'WBC', 'Platelet'])
    min_cell_size: int = 10  # minimum pixels
    max_cell_size: int = 500  # maximum pixels
    enable_morphology_analysis: bool = True
    save_annotated_images: bool = True
    enable_batch_processing: bool = True
    max_batch_size: int = 50
    quality_threshold: float = 0.7  # minimum quality score
    enable_preprocessing: bool = True

class Config:
    """
    Main configuration class that manages all application settings
    """
    
    def __init__(self, config_file: Optional[str] = None, environment: str = "development"):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to JSON config file
            environment: Environment name (development, production, testing)
        """
        self.environment = environment
        self.config_file = config_file
        
        # Initialize configuration sections
        self.model = ModelConfig()
        self.file = FileConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        self.analysis = AnalysisConfig()
        
        # Load configuration
        self._load_environment_variables()
        self._load_config_file()
        self._apply_environment_specific_settings()
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {environment}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            env_files = ['.env', f'.env.{self.environment}', '.env.local']
            for env_file in env_files:
                if os.path.exists(env_file):
                    load_dotenv(env_file, override=True)
                    logger.info(f"Loaded environment file: {env_file}")
        
        # Model configuration
        self.model.yolo_model_path = os.getenv('YOLO_MODEL_PATH', self.model.yolo_model_path)
        self.model.classifier_model_path = os.getenv('CLASSIFIER_MODEL_PATH', self.model.classifier_model_path)
        self.model.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', self.model.confidence_threshold))
        self.model.device = os.getenv('MODEL_DEVICE', self.model.device)
        self.model.batch_size = int(os.getenv('MODEL_BATCH_SIZE', self.model.batch_size))
        self.model.input_size = int(os.getenv('MODEL_INPUT_SIZE', self.model.input_size))
        
        # File configuration
        self.file.upload_folder = os.getenv('UPLOAD_FOLDER', self.file.upload_folder)
        self.file.results_folder = os.getenv('RESULTS_FOLDER', self.file.results_folder)
        self.file.max_file_size = int(os.getenv('MAX_FILE_SIZE', self.file.max_file_size))
        allowed_ext = os.getenv('ALLOWED_EXTENSIONS')
        if allowed_ext:
            self.file.allowed_extensions = [ext.strip() for ext in allowed_ext.split(',')]
        
        # Database configuration
        self.database.sqlite_path = os.getenv('SQLITE_PATH', self.database.sqlite_path)
        self.database.firebase_project_id = os.getenv('FIREBASE_PROJECT_ID')
        self.database.firebase_credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
        self.database.firebase_storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
        
        # API configuration
        self.api.host = os.getenv('FLASK_HOST', self.api.host)
        self.api.port = int(os.getenv('FLASK_PORT', self.api.port))
        self.api.debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        self.api.secret_key = os.getenv('SECRET_KEY', self.api.secret_key)
        cors_origins = os.getenv('CORS_ORIGINS')
        if cors_origins:
            self.api.cors_origins = [origin.strip() for origin in cors_origins.split(',')]
        
        # Security configuration
        self.security.jwt_secret_key = os.getenv('JWT_SECRET_KEY')
        self.security.api_key_required = os.getenv('API_KEY_REQUIRED', 'False').lower() == 'true'
        self.security.secure_cookies = os.getenv('SECURE_COOKIES', 'False').lower() == 'true'
        
        # Logging configuration
        self.logging.level = os.getenv('LOG_LEVEL', self.logging.level)
        self.logging.log_file = os.getenv('LOG_FILE', self.logging.log_file)
        
        # Performance configuration
        self.performance.worker_processes = int(os.getenv('WORKER_PROCESSES', self.performance.worker_processes))
        self.performance.memory_limit_mb = int(os.getenv('MEMORY_LIMIT_MB', self.performance.memory_limit_mb))
        self.performance.enable_caching = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
    
    def _load_config_file(self):
        """Load configuration from JSON file if provided"""
        if not self.config_file or not os.path.exists(self.config_file):
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            logger.info(f"Configuration loaded from file: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file}: {e}")
    
    def _apply_environment_specific_settings(self):
        """Apply environment-specific configuration overrides"""
        
        if self.environment == "production":
            # Production settings
            self.api.debug = False
            self.security.secure_cookies = True
            self.security.api_key_required = True
            self.logging.level = "WARNING"
            self.performance.enable_profiling = False
            
            # Ensure secure secret key
            if self.api.secret_key == "dev-secret-key-change-in-production":
                raise ValueError("Production environment requires a secure SECRET_KEY")
        
        elif self.environment == "testing":
            # Testing settings
            self.api.debug = True
            self.database.sqlite_path = ":memory:"  # In-memory database for tests
            self.file.upload_folder = "test_uploads"
            self.logging.level = "DEBUG"
            self.performance.enable_caching = False
        
        elif self.environment == "development":
            # Development settings
            self.api.debug = True
            self.logging.level = "DEBUG"
            self.performance.enable_profiling = True
            self.security.api_key_required = False
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []
        warnings = []
        
        # Validate model paths
        if not os.path.exists(self.model.yolo_model_path):
            warnings.append(f"YOLO model not found: {self.model.yolo_model_path}")
        
        if self.model.classifier_model_path and not os.path.exists(self.model.classifier_model_path):
            warnings.append(f"Classifier model not found: {self.model.classifier_model_path}")
        
        # Validate thresholds
        if not (0.0 <= self.model.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        # Validate file size limits
        if self.file.max_file_size <= 0:
            errors.append("Max file size must be positive")
        
        # Validate API configuration
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        # Validate Firebase configuration
        if self.database.firebase_project_id:
            if self.database.firebase_credentials_path and not os.path.exists(self.database.firebase_credentials_path):
                warnings.append(f"Firebase credentials file not found: {self.database.firebase_credentials_path}")
        
        # Validate security settings
        if self.environment == "production":
            if len(self.api.secret_key) < 32:
                errors.append("Secret key should be at least 32 characters in production")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Configuration validation failed: {errors}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.file.upload_folder,
            self.file.results_folder,
            self.file.temp_folder,
            self.file.models_folder,
            os.path.dirname(self.logging.log_file),
            os.path.dirname(self.database.sqlite_path) if self.database.sqlite_path != ":memory:" else None
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                except Exception as e:
                    logger.error(f"Failed to create directory {directory}: {e}")
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask-specific configuration dictionary"""
        return {
            'SECRET_KEY': self.api.secret_key,
            'DEBUG': self.api.debug,
            'MAX_CONTENT_LENGTH': self.api.max_content_length,
            'UPLOAD_FOLDER': self.file.upload_folder,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': self.api.debug,
        }
    
    def get_database_url(self) -> str:
        """Get appropriate database URL based on configuration"""
        if self.database.firebase_project_id:
            return f"firebase://{self.database.firebase_project_id}"
        else:
            return f"sqlite:///{self.database.sqlite_path}"
    
    def get_allowed_extensions(self) -> set:
        """Get set of allowed file extensions"""
        return {ext.lower() for ext in self.file.allowed_extensions}
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == "testing"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            'environment': self.environment,
            'model': {
                'yolo_model_path': self.model.yolo_model_path,
                'confidence_threshold': self.model.confidence_threshold,
                'device': self.model.device,
                'input_size': self.model.input_size,
                'batch_size': self.model.batch_size
            },
            'file': {
                'max_file_size': self.file.max_file_size,
                'allowed_extensions': self.file.allowed_extensions,
                'upload_folder': self.file.upload_folder
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'cors_origins': self.api.cors_origins
            },
            'analysis': {
                'default_cell_types': self.analysis.default_cell_types,
                'enable_morphology_analysis': self.analysis.enable_morphology_analysis,
                'max_batch_size': self.analysis.max_batch_size
            }
        }
    
    def save_config(self, filepath: str):
        """Save current configuration to JSON file"""
        try:
            config_dict = self.to_dict()
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Configuration saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for section_name, section_updates in updates.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_updates.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                        logger.info(f"Updated {section_name}.{key} = {value}")
        
        # Re-validate after updates
        self._validate_configuration()

class ConfigManager:
    """
    Configuration manager for handling multiple configuration instances
    """
    
    _instances = {}
    _default_config = None
    
    @classmethod
    def get_config(cls, environment: str = None, config_file: str = None) -> Config:
        """
        Get configuration instance (singleton per environment)
        
        Args:
            environment: Environment name
            config_file: Optional config file path
            
        Returns:
            Config instance
        """
        if environment is None:
            environment = os.getenv('FLASK_ENV', 'development')
        
        cache_key = f"{environment}:{config_file or 'default'}"
        
        if cache_key not in cls._instances:
            cls._instances[cache_key] = Config(config_file, environment)
            
            # Set as default if first instance
            if cls._default_config is None:
                cls._default_config = cls._instances[cache_key]
        
        return cls._instances[cache_key]
    
    @classmethod
    def get_default_config(cls) -> Config:
        """Get default configuration instance"""
        if cls._default_config is None:
            cls._default_config = cls.get_config()
        return cls._default_config
    
    @classmethod
    def reload_config(cls, environment: str = None):
        """Reload configuration for given environment"""
        if environment is None:
            environment = os.getenv('FLASK_ENV', 'development')
        
        # Clear cached instances for this environment
        keys_to_remove = [key for key in cls._instances.keys() if key.startswith(f"{environment}:")]
        for key in keys_to_remove:
            del cls._instances[key]
        
        # Create new instance
        return cls.get_config(environment)

# Convenience functions
def get_config(environment: str = None) -> Config:
    """Get configuration instance"""
    return ConfigManager.get_config(environment)

def get_flask_config(environment: str = None) -> Dict[str, Any]:
    """Get Flask configuration dictionary"""
    return get_config(environment).get_flask_config()

def setup_logging(config: Config = None):
    """Setup logging based on configuration"""
    if config is None:
        config = get_config()
    
    # Create logs directory
    log_dir = os.path.dirname(config.logging.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_format = config.logging.format
    log_level = getattr(logging, config.logging.level.upper())
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup handlers
    handlers = []
    
    if config.logging.console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    if config.logging.file_handler:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.logging.log_file,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging configured - Level: {config.logging.level}, File: {config.logging.log_file}")

# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config = get_config('development')
    
    print("Configuration Summary:")
    print(f"Environment: {config.environment}")
    print(f"API Host: {config.api.host}:{config.api.port}")
    print(f"Debug Mode: {config.api.debug}")
    print(f"YOLO Model: {config.model.yolo_model_path}")
    print(f"Upload Folder: {config.file.upload_folder}")
    
    # Test configuration validation
    try:
        config._validate_configuration()
        print("✅ Configuration validation passed")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
    
    # Test Flask config
    flask_config = config.get_flask_config()
    print(f"Flask Config Keys: {list(flask_config.keys())}")
    
    # Test configuration saving
    config.save_config('config_export.json')
    print("Configuration exported to config_export.json")