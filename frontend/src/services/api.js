import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens (if needed)
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API service functions
export const apiService = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/api/health');
      return response.data;
    } catch (error) {
      throw new Error('Health check failed');
    }
  },

  // Single image analysis
  analyzeSingleImage: async (imageFile, onProgress = null) => {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };

      if (onProgress) {
        config.onUploadProgress = (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(progress);
        };
      }

      const response = await api.post('/api/analyze', formData, config);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Analysis failed');
    }
  },

  // Batch image analysis
  analyzeBatchImages: async (imageFiles, onProgress = null) => {
    try {
      const formData = new FormData();
      imageFiles.forEach(file => {
        formData.append('images', file);
      });

      const config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };

      if (onProgress) {
        config.onUploadProgress = (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(progress);
        };
      }

      const response = await api.post('/api/batch-analyze', formData, config);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Batch analysis failed');
    }
  },

  // Get analysis history (if implemented)
  getAnalysisHistory: async (limit = 10) => {
    try {
      const response = await api.get(`/api/history?limit=${limit}`);
      return response.data;
    } catch (error) {
      throw new Error('Failed to fetch analysis history');
    }
  },

  // Get analysis by ID (if implemented)
  getAnalysisById: async (analysisId) => {
    try {
      const response = await api.get(`/api/analysis/${analysisId}`);
      return response.data;
    } catch (error) {
      throw new Error('Failed to fetch analysis details');
    }
  },

  // Export analysis results (if implemented)
  exportAnalysis: async (analysisId, format = 'json') => {
    try {
      const response = await api.get(`/api/export/${analysisId}?format=${format}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      throw new Error('Failed to export analysis');
    }
  },
};

export default apiService;