# backend/utils/firebase_config.py
"""
Firebase Configuration and Service Integration

This module provides Firebase integration for the Blood Cell Analyzer,
including Firestore database operations, Cloud Storage, and Authentication.

Features:
- Firestore database operations for analysis results
- Cloud Storage for image uploads
- Authentication management
- Real-time analytics and metrics collection
- Batch operations and data synchronization
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import time
from pathlib import Path

# Firebase imports with error handling
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage, auth
    from google.cloud.firestore import Client as FirestoreClient
    from google.cloud.storage import Client as StorageClient
    FIREBASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Firebase dependencies not available: {e}")
    firebase_admin = None
    credentials = None
    firestore = None
    storage = None
    auth = None
    FirestoreClient = None
    StorageClient = None
    FIREBASE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FirebaseConfig:
    """Firebase configuration management"""
    
    def __init__(self):
        self.project_id = os.getenv('FIREBASE_PROJECT_ID')
        self.credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
        self.storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET') or f"{self.project_id}.appspot.com"
        self.database_url = os.getenv('FIREBASE_DATABASE_URL')
        
        # Collection names
        self.collections = {
            'analyses': 'blood_cell_analyses',
            'users': 'users',
            'metrics': 'system_metrics',
            'sessions': 'analysis_sessions',
            'feedback': 'user_feedback'
        }
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate Firebase configuration"""
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase SDK not available")
            return False
        
        if not self.project_id:
            logger.error("FIREBASE_PROJECT_ID environment variable not set")
            return False
        
        if self.credentials_path and not os.path.exists(self.credentials_path):
            logger.error(f"Firebase credentials file not found: {self.credentials_path}")
            return False
        
        logger.info("Firebase configuration validated successfully")
        return True
    
    def get_credentials(self):
        """Get Firebase credentials"""
        if self.credentials_path and os.path.exists(self.credentials_path):
            return credentials.Certificate(self.credentials_path)
        else:
            # Use default credentials (for production environments)
            return credentials.ApplicationDefault()

def initialize_firebase() -> tuple[Optional[FirestoreClient], Optional[Any]]:
    """
    Initialize Firebase Admin SDK
    
    Returns:
        Tuple of (firestore_client, storage_bucket) or (None, None) if failed
    """
    if not FIREBASE_AVAILABLE:
        logger.warning("Firebase not available - using mock services")
        return None, None
    
    try:
        config = FirebaseConfig()
        
        # Initialize Firebase app if not already initialized
        if not firebase_admin._apps:
            cred = config.get_credentials()
            firebase_admin.initialize_app(cred, {
                'projectId': config.project_id,
                'storageBucket': config.storage_bucket
            })
            logger.info(f"Firebase initialized for project: {config.project_id}")
        
        # Get Firestore client
        db = firestore.client()
        
        # Get Storage bucket
        bucket = storage.bucket()
        
        return db, bucket
        
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        return None, None

class FirebaseService:
    """
    Comprehensive Firebase service for blood cell analysis platform
    """
    
    def __init__(self):
        self.config = FirebaseConfig()
        self.db, self.bucket = initialize_firebase()
        self.initialized = self.db is not None
        
        if not self.initialized:
            logger.warning("Firebase service running in mock mode")
    
    # ============================================================================
    # ANALYSIS RESULTS MANAGEMENT
    # ============================================================================
    
    def save_analysis_result(self, analysis_data: Dict[str, Any], user_id: str = 'anonymous') -> Optional[str]:
        """
        Save analysis results to Firestore
        
        Args:
            analysis_data: Analysis results dictionary
            user_id: User identifier
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.initialized:
            logger.warning("Firebase not available - analysis not saved")
            return None
        
        try:
            # Prepare document data
            doc_data = {
                'user_id': user_id,
                'timestamp': datetime.now(timezone.utc),
                'analysis_id': analysis_data.get('analysis_id', f"analysis_{int(time.time())}"),
                'cell_counts': analysis_data.get('cell_counts', {}),
                'processing_time': analysis_data.get('processing_time', 0),
                'confidence_score': analysis_data.get('confidence_score', 0),
                'total_cells_detected': analysis_data.get('total_cells_detected', 0),
                'image_metadata': {
                    'filename': analysis_data.get('filename'),
                    'image_size': analysis_data.get('image_size'),
                    'format': analysis_data.get('image_format')
                },
                'model_info': {
                    'yolo_version': analysis_data.get('yolo_version', 'YOLOv5'),
                    'classifier_version': analysis_data.get('classifier_version', 'v1.0'),
                    'device_used': analysis_data.get('device_used', 'cpu')
                },
                'status': 'completed',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            
            # Add detected objects if available (limit for storage efficiency)
            detected_objects = analysis_data.get('detected_objects', [])
            if detected_objects:
                doc_data['detected_objects'] = detected_objects[:100]  # Limit to 100 objects
                doc_data['total_objects_stored'] = len(detected_objects[:100])
                doc_data['total_objects_detected'] = len(detected_objects)
            
            # Save to Firestore
            collection_ref = self.db.collection(self.config.collections['analyses'])
            doc_ref = collection_ref.add(doc_data)
            
            document_id = doc_ref[1].id
            logger.info(f"Analysis saved with ID: {document_id}")
            
            # Update user statistics
            self._update_user_stats(user_id, analysis_data)
            
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
            return None
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis by document ID
        
        Args:
            analysis_id: Firestore document ID
            
        Returns:
            Analysis data or None if not found
        """
        if not self.initialized:
            return None
        
        try:
            doc_ref = self.db.collection(self.config.collections['analyses']).document(analysis_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            else:
                logger.warning(f"Analysis {analysis_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve analysis {analysis_id}: {e}")
            return None
    
    def get_user_analyses(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get user's analysis history
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of analysis results
        """
        if not self.initialized:
            return []
        
        try:
            query = self.db.collection(self.config.collections['analyses'])\
                .where('user_id', '==', user_id)\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .offset(offset)
            
            results = []
            for doc in query.stream():
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)
            
            logger.info(f"Retrieved {len(results)} analyses for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve user analyses: {e}")
            return []
    
    def delete_analysis(self, analysis_id: str, user_id: str) -> bool:
        """
        Delete an analysis result
        
        Args:
            analysis_id: Document ID to delete
            user_id: User ID for authorization
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            doc_ref = self.db.collection(self.config.collections['analyses']).document(analysis_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.warning(f"Analysis {analysis_id} not found for deletion")
                return False
            
            # Verify ownership
            doc_data = doc.to_dict()
            if doc_data.get('user_id') != user_id:
                logger.warning(f"User {user_id} not authorized to delete analysis {analysis_id}")
                return False
            
            # Delete the document
            doc_ref.delete()
            logger.info(f"Analysis {analysis_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete analysis {analysis_id}: {e}")
            return False
    
    # ============================================================================
    # IMAGE STORAGE MANAGEMENT
    # ============================================================================
    
    def upload_image(self, image_data: bytes, filename: str, user_id: str = 'anonymous') -> Optional[str]:
        """
        Upload image to Firebase Storage
        
        Args:
            image_data: Image binary data
            filename: Original filename
            user_id: User identifier
            
        Returns:
            Public URL of uploaded image or None if failed
        """
        if not self.initialized or not self.bucket:
            logger.warning("Firebase Storage not available")
            return None
        
        try:
            # Generate unique filename
            timestamp = int(time.time())
            safe_filename = self._sanitize_filename(filename)
            storage_path = f"images/{user_id}/{timestamp}_{safe_filename}"
            
            # Upload to Storage
            blob = self.bucket.blob(storage_path)
            blob.upload_from_string(image_data, content_type='image/jpeg')
            
            # Make publicly accessible (optional - adjust based on requirements)
            blob.make_public()
            
            logger.info(f"Image uploaded to: {storage_path}")
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            return None
    
    def delete_image(self, image_url: str) -> bool:
        """
        Delete image from Firebase Storage
        
        Args:
            image_url: Public URL of image to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not self.bucket:
            return False
        
        try:
            # Extract blob name from URL
            blob_name = self._extract_blob_name(image_url)
            if not blob_name:
                return False
            
            blob = self.bucket.blob(blob_name)
            blob.delete()
            
            logger.info(f"Image deleted: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            return False
    
    # ============================================================================
    # USER MANAGEMENT
    # ============================================================================
    
    def create_or_update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """
        Create or update user profile
        
        Args:
            user_id: User identifier
            user_data: User profile data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            user_doc = {
                'user_id': user_id,
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'role': user_data.get('role', 'user'),
                'created_at': user_data.get('created_at', datetime.now(timezone.utc)),
                'updated_at': datetime.now(timezone.utc),
                'last_login': datetime.now(timezone.utc),
                'total_analyses': 0,
                'settings': user_data.get('settings', {})
            }
            
            doc_ref = self.db.collection(self.config.collections['users']).document(user_id)
            doc_ref.set(user_doc, merge=True)
            
            logger.info(f"User profile updated: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile data
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile data or None if not found
        """
        if not self.initialized:
            return None
        
        try:
            doc_ref = self.db.collection(self.config.collections['users']).document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    def _update_user_stats(self, user_id: str, analysis_data: Dict[str, Any]):
        """Update user statistics after analysis"""
        try:
            user_ref = self.db.collection(self.config.collections['users']).document(user_id)
            user_ref.update({
                'total_analyses': firestore.Increment(1),
                'last_analysis': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            })
        except Exception as e:
            logger.warning(f"Failed to update user stats: {e}")
    
    # ============================================================================
    # ANALYTICS AND METRICS
    # ============================================================================
    
    def record_system_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """
        Record system performance metrics
        
        Args:
            metrics_data: Metrics to record
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            metrics_doc = {
                'timestamp': datetime.now(timezone.utc),
                'processing_time': metrics_data.get('processing_time'),
                'memory_usage': metrics_data.get('memory_usage'),
                'cpu_usage': metrics_data.get('cpu_usage'),
                'gpu_usage': metrics_data.get('gpu_usage'),
                'model_confidence': metrics_data.get('model_confidence'),
                'error_count': metrics_data.get('error_count', 0),
                'success_count': metrics_data.get('success_count', 1)
            }
            
            self.db.collection(self.config.collections['metrics']).add(metrics_doc)
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
            return False
    
    def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics summary for specified period
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics summary
        """
        if not self.initialized:
            return {}
        
        try:
            from datetime import timedelta
            
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Query analyses in date range
            analyses_query = self.db.collection(self.config.collections['analyses'])\
                .where('timestamp', '>=', start_date)\
                .where('timestamp', '<=', end_date)
            
            total_analyses = 0
            total_cells = 0
            total_processing_time = 0
            cell_type_counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0}
            
            for doc in analyses_query.stream():
                data = doc.to_dict()
                total_analyses += 1
                total_cells += data.get('total_cells_detected', 0)
                total_processing_time += data.get('processing_time', 0)
                
                # Count cell types
                cell_counts = data.get('cell_counts', {})
                for cell_type, count in cell_counts.items():
                    if cell_type in cell_type_counts:
                        cell_type_counts[cell_type] += count
            
            # Calculate averages
            avg_processing_time = total_processing_time / total_analyses if total_analyses > 0 else 0
            avg_cells_per_analysis = total_cells / total_analyses if total_analyses > 0 else 0
            
            return {
                'period_days': days,
                'total_analyses': total_analyses,
                'total_cells_detected': total_cells,
                'average_processing_time': avg_processing_time,
                'average_cells_per_analysis': avg_cells_per_analysis,
                'cell_type_distribution': cell_type_counts,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
    
    # ============================================================================
    # BATCH OPERATIONS
    # ============================================================================
    
    def batch_save_analyses(self, analyses_data: List[Dict[str, Any]]) -> List[str]:
        """
        Save multiple analyses in a batch operation
        
        Args:
            analyses_data: List of analysis data dictionaries
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            return []
        
        try:
            batch = self.db.batch()
            doc_ids = []
            
            collection_ref = self.db.collection(self.config.collections['analyses'])
            
            for analysis_data in analyses_data:
                doc_ref = collection_ref.document()
                doc_data = {
                    'timestamp': datetime.now(timezone.utc),
                    **analysis_data,
                    'batch_processed': True
                }
                batch.set(doc_ref, doc_data)
                doc_ids.append(doc_ref.id)
            
            # Commit batch
            batch.commit()
            logger.info(f"Batch saved {len(doc_ids)} analyses")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Batch save failed: {e}")
            return []
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for storage"""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        return sanitized[:100]  # Limit length
    
    def _extract_blob_name(self, url: str) -> Optional[str]:
        """Extract blob name from Firebase Storage URL"""
        try:
            # Parse Firebase Storage URL to get blob name
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            path_parts = parsed.path.split('/')
            if len(path_parts) >= 4 and path_parts[1] == 'v0' and path_parts[2] == 'b':
                # Extract everything after /o/
                o_index = path_parts.index('o')
                blob_name = '/'.join(path_parts[o_index + 1:])
                return urllib.parse.unquote(blob_name)
            return None
        except Exception:
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Firebase services
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'firebase_available': FIREBASE_AVAILABLE,
            'service_initialized': self.initialized,
            'firestore_connected': False,
            'storage_connected': False,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if self.initialized:
            try:
                # Test Firestore connection
                self.db.collection('health_check').limit(1).get()
                health_status['firestore_connected'] = True
            except Exception as e:
                logger.warning(f"Firestore health check failed: {e}")
            
            try:
                # Test Storage connection
                if self.bucket:
                    list(self.bucket.list_blobs(max_results=1))
                    health_status['storage_connected'] = True
            except Exception as e:
                logger.warning(f"Storage health check failed: {e}")
        
        return health_status
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """
        Cleanup old analysis data
        
        Args:
            days_old: Delete data older than this many days
            
        Returns:
            Cleanup statistics
        """
        if not self.initialized:
            return {'deleted_analyses': 0, 'deleted_images': 0}
        
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            # Query old analyses
            old_analyses = self.db.collection(self.config.collections['analyses'])\
                .where('timestamp', '<', cutoff_date)\
                .stream()
            
            deleted_count = 0
            batch = self.db.batch()
            
            for doc in old_analyses:
                batch.delete(doc.reference)
                deleted_count += 1
                
                # Commit in batches of 500 (Firestore limit)
                if deleted_count % 500 == 0:
                    batch.commit()
                    batch = self.db.batch()
            
            # Commit remaining
            if deleted_count % 500 != 0:
                batch.commit()
            
            logger.info(f"Cleaned up {deleted_count} old analyses")
            return {'deleted_analyses': deleted_count, 'deleted_images': 0}
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {'deleted_analyses': 0, 'deleted_images': 0}

# Mock Firebase Service for development/testing
class MockFirebaseService:
    """Mock Firebase service for testing when Firebase is not available"""
    
    def __init__(self):
        self.data_store = {}
        self.next_id = 1
        logger.info("Using Mock Firebase Service")
    
    def save_analysis_result(self, analysis_data: Dict[str, Any], user_id: str = 'anonymous') -> Optional[str]:
        doc_id = f"mock_{self.next_id}"
        self.next_id += 1
        self.data_store[doc_id] = {
            'id': doc_id,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc),
            **analysis_data
        }
        logger.info(f"Mock: Saved analysis {doc_id}")
        return doc_id
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        return self.data_store.get(analysis_id)
    
    def get_user_analyses(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        user_analyses = [data for data in self.data_store.values() if data.get('user_id') == user_id]
        return user_analyses[offset:offset + limit]
    
    def health_check(self) -> Dict[str, Any]:
        return {
            'firebase_available': False,
            'service_initialized': True,
            'firestore_connected': False,
            'storage_connected': False,
            'mock_mode': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def record_system_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        logger.info(f"Mock: Recorded metrics: {metrics_data}")
        return True
    
    def upload_image(self, image_data: bytes, filename: str, user_id: str = 'anonymous') -> Optional[str]:
        mock_url = f"mock://storage/{user_id}/{filename}"
        logger.info(f"Mock: Uploaded image to {mock_url}")
        return mock_url

# Factory function to get appropriate service
def get_firebase_service() -> Union[FirebaseService, MockFirebaseService]:
    """
    Get Firebase service instance (real or mock based on availability)
    
    Returns:
        FirebaseService or MockFirebaseService instance
    """
    if FIREBASE_AVAILABLE:
        return FirebaseService()
    else:
        return MockFirebaseService()

# Example usage and testing
if __name__ == "__main__":
    # Test Firebase service
    service = get_firebase_service()
    
    # Test health check
    health = service.health_check()
    print(f"Firebase Health: {health}")
    
    # Test saving analysis
    sample_analysis = {
        'analysis_id': 'test_analysis_1',
        'cell_counts': {'RBC': 100, 'WBC': 15, 'Platelet': 25},
        'processing_time': 1.5,
        'confidence_score': 0.92
    }
    
    doc_id = service.save_analysis_result(sample_analysis, 'test_user')
    print(f"Saved analysis with ID: {doc_id}")
    
    # Test retrieval
    if doc_id:
        retrieved = service.get_analysis_by_id(doc_id)
        print(f"Retrieved analysis: {retrieved}")