import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import '../styles/ImageUpload.css';

const ImageUpload = ({ onAnalysisComplete, loading, setLoading }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    setSelectedFiles(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.tiff', '.bmp']
    },
    multiple: true,
    maxSize: 10485760 // 10MB
  });

  const analyzeImages = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select at least one image');
      return;
    }

    setLoading(true);
    setUploadProgress(0);

    try {
      if (selectedFiles.length === 1) {
        // Single image analysis
        const formData = new FormData();
        formData.append('image', selectedFiles[0]);

        const response = await axios.post('/api/analyze', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(progress);
          },
        });

        onAnalysisComplete(response.data);
      } else {
        // Batch analysis
        const formData = new FormData();
        selectedFiles.forEach(file => {
          formData.append('images', file);
        });

        const response = await axios.post('/api/batch-analyze', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(progress);
          },
        });

        onAnalysisComplete(response.data);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const removeFile = (index) => {
    setSelectedFiles(files => files.filter((_, i) => i !== index));
  };

  return (
    <div className="image-upload-container">
      <div className="upload-section">
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <div className="dropzone-content">
            <div className="upload-icon">📁</div>
            <h3>Upload Blood Smear Images</h3>
            <p>
              {isDragActive
                ? 'Drop the images here...'
                : 'Drag & drop images here, or click to select files'}
            </p>
            <p className="file-info">
              Supports: JPEG, PNG, TIFF, BMP (Max 10MB each)
            </p>
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div className="selected-files">
            <h4>Selected Files ({selectedFiles.length})</h4>
            <div className="file-list">
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <div className="file-info">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="remove-btn"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="analysis-controls">
          <button
            onClick={analyzeImages}
            disabled={loading || selectedFiles.length === 0}
            className="analyze-btn"
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing... {uploadProgress}%
              </>
            ) : (
              <>
                🔬 Analyze Blood Cells
                {selectedFiles.length > 1 && ` (${selectedFiles.length} images)`}
              </>
            )}
          </button>
        </div>

        {loading && (
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        )}
      </div>

      <div className="info-panel">
        <h3>🩸 How it works</h3>
        <div className="info-steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Upload Images</h4>
              <p>Select blood smear microscopy images</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>AI Analysis</h4>
              <p>YOLOv5 model detects and classifies cells</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Get Results</h4>
              <p>View cell counts, percentages, and annotations</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;