import pytest
import os
import tempfile
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def temp_upload_dir():
    """Create temporary upload directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    import shutil
    shutil.rmtree(temp_dir)
